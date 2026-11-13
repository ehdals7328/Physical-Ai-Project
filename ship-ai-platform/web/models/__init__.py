# web/models/__init__.py
from .schema import (
    db,
    User,
    WorkOrder,
    WorkInstruction,
    AgentExecution,
    AgentChatHistory,
    DocumentMetadata,
    VisionAnalysis,
    VisionDetection,
    VisionInspection,
    RobotExecution,
    CommunityBoard,
    ErpCompany,
    ErpBranch,
    ErpBlockMaster,
    ErpBomInventory,
    ErpWorkerCert,
    ErpVehiclesEquipment,
    MesWorkOrderDispatch,
)
from .partner_b2b import (
    B2BCompanyLocation,
    B2BDaehanMaterial,
    B2BDonghaeWorker,
    B2BHankookEquipment,
    B2BSalesOrder,
)