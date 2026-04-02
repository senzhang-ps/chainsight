# models.py
# 编排器数据模型定义
#
# 定义编排器使用的数据类和结构

from dataclasses import dataclass


@dataclass
class DeploymentUID:
    """用于部署跟踪的唯一标识符"""

    material: str
    sending: str
    receiving: str
    planned_deploy_date: str  # YYYY-MM-DD format
    demand_element: str
    sequence: int  # Auto-incrementing sequence for uniqueness

    def to_string(self) -> str:
        """转换为字符串表示以便跟踪"""
        mat = self.material
        snd = self.sending
        rcv = self.receiving
        pdd = self.planned_deploy_date
        de = self.demand_element
        seq = self.sequence
        return (
            f"{mat}|{snd}|{rcv}|{pdd}|{de}|{seq:06d}"
        )

    @classmethod
    def from_string(cls, uid_str: str) -> 'DeploymentUID':
        """从字符串表示解析"""
        parts = uid_str.split('|')
        return cls(
            material=parts[0],
            sending=parts[1],
            receiving=parts[2],
            planned_deploy_date=parts[3],
            demand_element=parts[4],
            sequence=int(parts[5]),
        )
