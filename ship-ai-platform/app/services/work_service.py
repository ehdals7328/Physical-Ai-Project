from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from app.extensions import db
from app.models.ship_user import (
    Ship,
    ShipBlock,
    WeldJoint,
    WeldingProcedure,
    WorkOrder,
    WorkOrderJoint,
)


def _optional_text(value):
    return str(value).strip() if value is not None and str(value).strip() else None


def _optional_float(value, label: str):
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}은 숫자로 입력해 주세요.") from exc
    if number < 0:
        raise ValueError(f"{label}은 0 이상이어야 합니다.")
    return number


def create_work_order(user, payload: dict) -> WorkOrder:
    ship_code = (payload.get("ship_code") or "").strip().upper()
    block_code = (payload.get("block_code") or "").strip().upper()
    joint_code = (payload.get("joint_code") or "").strip().upper()
    title = (payload.get("work_title") or "").strip()
    if not all((ship_code, block_code, joint_code, title)):
        raise ValueError("선박 코드, 블록 코드, 용접부 코드, 작업명은 필수입니다.")

    plate_thickness = _optional_float(payload.get("plate_thickness_mm"), "모재 두께")
    design_length = _optional_float(payload.get("design_length_mm"), "용접 길이")
    revision = (payload.get("drawing_revision") or "A").strip().upper()

    # 비관리자는 계정에 등록된 사업장 범위 안에서만 작업을 생성한다.
    requested_branch = _optional_text(payload.get("erp_branch_code"))
    if user.role_code != "SYS_ADMIN":
        account_branch = _optional_text(user.erp_branch_code)
        if requested_branch and account_branch and requested_branch.upper() != account_branch.upper():
            raise PermissionError("계정에 지정된 사업장 외의 작업은 등록할 수 없습니다.")
        if requested_branch and not account_branch:
            raise PermissionError("계정에 사업장 코드가 등록되어 있지 않습니다. 관리자에게 문의하세요.")
        requested_branch = account_branch

    wps = None
    wps_code = _optional_text(payload.get("wps_code"))
    if wps_code:
        wps_revision = (payload.get("wps_revision") or "").strip().upper()
        if not wps_revision:
            raise ValueError("WPS 코드를 입력한 경우 WPS 개정 번호도 입력해야 합니다.")
        wps = WeldingProcedure.query.filter_by(wps_code=wps_code, revision_no=wps_revision).first()
        if wps is None:
            raise ValueError("입력한 WPS 코드와 개정 번호를 찾을 수 없습니다.")
        if wps.approval_status != "APPROVED":
            raise ValueError("승인 상태가 APPROVED인 WPS만 작업에 연결할 수 있습니다.")

    ship = Ship.query.filter_by(ship_code=ship_code).first()
    if ship is None:
        ship = Ship(ship_code=ship_code, ship_name=_optional_text(payload.get("ship_name")) or ship_code)
        db.session.add(ship)
        db.session.flush()

    block = ShipBlock.query.filter_by(ship_id=ship.ship_id, block_code=block_code).first()
    if block is not None and user.role_code != "SYS_ADMIN" and requested_branch:
        if block.erp_branch_code and block.erp_branch_code.upper() != requested_branch.upper():
            raise PermissionError("해당 블록은 계정에 지정된 사업장 범위에 속하지 않습니다.")
        if not block.erp_branch_code:
            block.erp_branch_code = requested_branch

    if block is None:
        block = ShipBlock(
            ship_id=ship.ship_id,
            block_code=block_code,
            block_name=_optional_text(payload.get("block_name")) or block_code,
            erp_branch_code=requested_branch,
            location_code=_optional_text(payload.get("location_code")),
            drawing_no=_optional_text(payload.get("drawing_no")),
            drawing_revision=revision,
        )
        db.session.add(block)
        db.session.flush()

    joint = WeldJoint.query.filter_by(
        block_id=block.block_id,
        joint_code=joint_code,
        drawing_revision=revision,
    ).first()
    if joint is None:
        base_material = _optional_text(payload.get("base_material_grade"))
        joint = WeldJoint(
            block_id=block.block_id,
            joint_code=joint_code,
            drawing_no=_optional_text(payload.get("drawing_no")),
            drawing_revision=revision,
            joint_type_code=(_optional_text(payload.get("joint_type_code")) or "BUTT").upper(),
            groove_type_code=_optional_text(payload.get("groove_type_code")),
            weld_position_code=_optional_text(payload.get("weld_position_code")),
            material_a_grade=base_material,
            material_b_grade=base_material,
            thickness_a_mm=plate_thickness,
            thickness_b_mm=plate_thickness,
            design_length_mm=design_length,
        )
        db.session.add(joint)
        db.session.flush()

    work_no = f"WO-{datetime.utcnow():%Y%m%d}-{uuid4().hex[:10].upper()}"
    order = WorkOrder(
        work_no=work_no,
        ship_id=ship.ship_id,
        block_id=block.block_id,
        work_type_code=(_optional_text(payload.get("work_type_code")) or "PRODUCTION").upper(),
        work_title=title[:200],
        work_description=_optional_text(payload.get("work_description")),
        process_code=_optional_text(payload.get("process_code")),
        base_material_grade=_optional_text(payload.get("base_material_grade")),
        plate_thickness_mm=plate_thickness,
        weld_position_code=_optional_text(payload.get("weld_position_code")),
        erp_branch_code=requested_branch,
        work_location_code=_optional_text(payload.get("location_code")),
        requested_by=user.user_id,
        priority_code=(_optional_text(payload.get("priority_code")) or "NORMAL").upper(),
        status_code="REQUESTED",
    )
    db.session.add(order)
    db.session.flush()

    relation = WorkOrderJoint(
        work_id=order.work_id,
        joint_id=joint.joint_id,
        wps_id=wps.wps_id if wps else None,
        sequence_no=1,
        planned_length_mm=design_length,
        status_code="PLANNED",
    )
    db.session.add(relation)
    db.session.commit()
    return order


def list_work_orders(user=None, limit: int = 100):
    query = WorkOrder.query
    if user is not None and user.role_code != "SYS_ADMIN":
        if user.erp_branch_code:
            query = query.filter(WorkOrder.erp_branch_code == user.erp_branch_code)
        else:
            # 회사/사업장 범위가 계정에 연결되지 않은 사용자는 본인이 등록한 작업만 조회한다.
            query = query.filter(WorkOrder.requested_by == user.user_id)
    rows = (
        query.order_by(WorkOrder.work_id.desc())
        .limit(max(1, min(limit, 300)))
        .all()
    )
    result = []
    for item in rows:
        block = db.session.get(ShipBlock, item.block_id) if item.block_id else None
        ship = db.session.get(Ship, item.ship_id) if item.ship_id else None
        result.append({
            "work_id": item.work_id,
            "work_no": item.work_no,
            "title": item.work_title,
            "ship_code": ship.ship_code if ship else "-",
            "block_code": block.block_code if block else "-",
            "process_code": item.process_code or "-",
            "status_code": item.status_code,
            "requested_at": item.requested_at.strftime("%Y-%m-%d %H:%M") if item.requested_at else "",
        })
    return result
