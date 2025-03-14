from .node_base import NodeBase
from .node_port import OutputPort, ParameterPort
from PySide6.QtWidgets import QGraphicsItem

class InputNode(NodeBase):
    def __init__(self, parent=None, path="input/"):
        node_dict = {
            "uid": None,
            "model_name": "Input Node",
            "output": {
                "Input Path": {
                    "connection": [],
                    "required": False
                }
            },
            "parameter": [
                {
                    "parameter": "folder_path",
                    "type": "str",
                    "default_value": f"{path}",
                    "max_value": None,
                    "min_value": None,
                    "current_value": f"{path}"
                }
            ],
            "scene_pos": [0, 0],
        }
        super().__init__(parent=parent, node_dict=node_dict, title="Input Node", width = 200, height = 70)

    def add_ports(self):
        self.output_port = OutputPort(text="Input Path")
        self.output_port.setParentItem(self)
        self.output_port.setPos(100, 30)
        self.output_port.setParentNode(self, index=0)
        self.output_ports = [self.output_port]

        self.parameter_port = ParameterPort(
            parameter="folder_path",
            type=str,
            default_value=self.node_dict["parameter"][0]["default_value"],
            max_value=None,
            min_value=None,
            current_value=self.node_dict["parameter"][0]["default_value"]
        )
        self.parameter_port.setParentItem(self)
        self.parameter_port.setPos(0, 50)
        self.parameter_port.setParentNode(self, index=0)
        self.addItem(self.parameter_port)
        self.height = 70

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            self.updatePortPositions()
        return super().itemChange(change, value)

    def updatePortPositions(self):
        for edge in self.edges:
            edge.updatePath()

    def addDownStreamNode(self, target_node_uid, output_port, target_port_idx):
        if output_port == self.output_port:
            self.node_dict["output"]["Input Path"]["connection"].append(target_node_uid + "_" + str(target_port_idx))

    def removeDownStreamNode(self, target_node_uid, output_port, target_port_idx):
        if output_port == self.output_port:
            self.node_dict["output"]["Input Path"]["connection"].remove(target_node_uid + "_" + str(target_port_idx))

    def to_dict(self):
        return self.node_dict