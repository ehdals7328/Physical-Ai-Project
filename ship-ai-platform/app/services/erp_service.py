from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from app.extensions import db
from app.models.ship_erp import (
    ErpBranch,
    ErpCompany,
    ErpEmployee,
    ErpEmployeeCertification,
    ErpEquipment,
    ErpEquipmentAssignment,
    ErpEquipmentType,
    ErpInventoryBalance,
    ErpMaterial,
    ErpMaterialLot,
    ErpPersonnelAssignment,
    ErpWarehouse,
)


def _as_float(value):
    return float(value) if value is not None else None


def list_branches(company_code: str | None = None, sys_admin: bool = False):
    query = ErpBranch.query.join(ErpCompany, ErpBranch.company_id == ErpCompany.company_id)
    if not sys_admin and company_code:
        query = query.filter(ErpCompany.company_code == company_code)
    elif not sys_admin:
        return []
    return query.order_by(ErpBranch.branch_name).all()


def branch_resources(branch_code: str, company_code: str | None, sys_admin: bool = False):
    branch_query = ErpBranch.query.filter_by(branch_code=branch_code)
    if not sys_admin:
        if not company_code:
            return None
        branch_query = (
            branch_query.join(ErpCompany, ErpBranch.company_id == ErpCompany.company_id)
            .filter(ErpCompany.company_code == company_code)
        )
    branch = branch_query.first()
    if branch is None:
        return None

    employees = (
        ErpEmployee.query.filter_by(branch_id=branch.branch_id, company_id=branch.company_id, employment_status="ACTIVE")
        .order_by(ErpEmployee.employee_name)
        .all()
    )
    equipment = (
        ErpEquipment.query.filter_by(branch_id=branch.branch_id, company_id=branch.company_id)
        .order_by(ErpEquipment.equipment_code)
        .all()
    )

    now = datetime.utcnow()
    active_personnel_assignments = ErpPersonnelAssignment.query.filter(
        ErpPersonnelAssignment.status_code.in_(["RESERVED", "CONFIRMED", "IN_USE"]),
        (ErpPersonnelAssignment.assigned_from.is_(None) | (ErpPersonnelAssignment.assigned_from <= now)),
        (ErpPersonnelAssignment.assigned_to.is_(None) | (ErpPersonnelAssignment.assigned_to >= now)),
    ).all()
    busy_employee_ids = {assignment.employee_id for assignment in active_personnel_assignments}

    active_equipment_assignments = ErpEquipmentAssignment.query.filter(
        ErpEquipmentAssignment.status_code.in_(["RESERVED", "CONFIRMED", "IN_USE"]),
        (ErpEquipmentAssignment.assigned_from.is_(None) | (ErpEquipmentAssignment.assigned_from <= now)),
        (ErpEquipmentAssignment.assigned_to.is_(None) | (ErpEquipmentAssignment.assigned_to >= now)),
    ).all()
    busy_equipment_ids = {assignment.equipment_id for assignment in active_equipment_assignments}

    inventories = (
        ErpInventoryBalance.query
        .join(ErpWarehouse, ErpInventoryBalance.warehouse_id == ErpWarehouse.warehouse_id)
        .join(ErpMaterialLot, ErpInventoryBalance.material_lot_id == ErpMaterialLot.material_lot_id)
        .join(ErpMaterial, ErpMaterialLot.material_id == ErpMaterial.material_id)
        .filter(
            ErpWarehouse.branch_id == branch.branch_id,
            ErpWarehouse.status_code == "ACTIVE",
            ErpMaterial.is_active == "Y",
        )
        .all()
    )

    employee_rows = []
    today = date.today()
    for employee in employees:
        cert_rows = ErpEmployeeCertification.query.filter_by(
            employee_id=employee.employee_id,
            verification_status="VERIFIED",
        ).all()
        valid_certs = [
            cert for cert in cert_rows
            if (cert.valid_from is None or cert.valid_from <= today)
            and (cert.valid_to is None or cert.valid_to >= today)
        ]
        certifications = [{
            "certification_code": cert.certification_code or "",
            "qualification_type": (cert.qualification_type or "").upper(),
            "standard_code": cert.standard_code or "",
            "welding_process_code": (cert.welding_process_code or "").upper(),
            "material_group": cert.material_group or "",
            "position_range": (cert.position_range or "").upper(),
            "thickness_min_mm": _as_float(cert.thickness_min_mm),
            "thickness_max_mm": _as_float(cert.thickness_max_mm),
            "valid_to": cert.valid_to.isoformat() if cert.valid_to else None,
        } for cert in valid_certs]
        employee_rows.append({
            "employee_code": employee.employee_code,
            "name": employee.employee_name,
            "role": employee.job_role_code or "-",
            "status": employee.employment_status,
            "valid_certifications": len(valid_certs),
            "certifications": certifications,
            "is_available": employee.employee_id not in busy_employee_ids,
        })

    equipment_rows = []
    for item in equipment:
        equipment_type = db.session.get(ErpEquipmentType, item.equipment_type_id)
        equipment_rows.append({
            "equipment_code": item.equipment_code,
            "name": item.equipment_name or item.equipment_code,
            "type": equipment_type.type_name if equipment_type else "설비",
            "type_code": equipment_type.type_code if equipment_type else "OTHER",
            "status": item.operation_status,
            "is_available": item.operation_status == "AVAILABLE" and item.equipment_id not in busy_equipment_ids,
        })

    inventory_rows = []
    for balance in inventories:
        lot = db.session.get(ErpMaterialLot, balance.material_lot_id)
        if lot is None:
            continue
        material = db.session.get(ErpMaterial, lot.material_id)
        if material is None:
            continue
        quantity = Decimal(balance.quantity or 0)
        reserved = Decimal(balance.reserved_quantity or 0)
        available = max(Decimal("0"), quantity - reserved) if lot.quality_status == "APPROVED" else Decimal("0")
        warehouse = db.session.get(ErpWarehouse, balance.warehouse_id)
        inventory_rows.append({
            "material_code": material.material_code,
            "material_name": material.material_name,
            "lot_code": lot.material_lot_code,
            "heat_no": lot.heat_no,
            "batch_no": lot.batch_no,
            "quality_status": lot.quality_status,
            "quantity": float(quantity),
            "reserved_quantity": float(reserved),
            "available_quantity": float(available),
            "unit": material.unit_code or "-",
            "warehouse_code": warehouse.warehouse_code if warehouse else "-",
            "warehouse_status": warehouse.status_code if warehouse else "INACTIVE",
            "is_shortage": available <= 0,
        })

    return {
        "branch_code": branch.branch_code,
        "branch_name": branch.branch_name,
        "workers": employee_rows,
        "equipment": equipment_rows,
        "inventory": inventory_rows,
        "available_worker_count": sum(1 for row in employee_rows if row["is_available"]),
        "available_equipment_count": sum(1 for row in equipment_rows if row["is_available"]),
    }


def inventory_summary(company_code: str | None, sys_admin: bool = False):
    output = []
    for branch in list_branches(company_code, sys_admin):
        snapshot = branch_resources(branch.branch_code, company_code, sys_admin)
        if snapshot:
            low_stock = [row for row in snapshot["inventory"] if row["is_shortage"]]
            output.append({
                "branch_code": branch.branch_code,
                "branch_name": branch.branch_name,
                "workers": len(snapshot["workers"]),
                "equipment": len(snapshot["equipment"]),
                "low_stock_count": len(low_stock),
            })
    return output
