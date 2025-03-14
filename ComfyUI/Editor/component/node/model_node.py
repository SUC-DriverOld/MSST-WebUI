from .node_base import NodeBase
from .node_port import InputPort, OutputPort, ParameterPort, BoolPort
import json
from PySide6.QtWidgets import QGraphicsItem

class ModelNode(NodeBase):
    def __init__(self, node_name, pos):
        self.node_name = node_name
        self.pos = pos if pos else [0, 0]
        self.set_node_dict()

        port_height = 20
        add_height_count = 0

        if "input" in self.node_dict: 
            add_height_count += len(self.node_dict["input"])
        if "output" in self.node_dict and len(self.node_dict["output"]) > 1:
            add_height_count += len(self.node_dict["output"]) - 1
        if "parameter" in self.node_dict:
            add_height_count += len(self.node_dict["parameter"])
        if "bool" in self.node_dict:
            add_height_count += len(self.node_dict["bool"])

        total_height = 30 + add_height_count * port_height    

        super().__init__(width=200, height=total_height, title="Model Node", subtitle=self.node_name)
        self.setPos(self.pos[0], self.pos[1])

    def set_node_dict(self):
        with open(f"./ComfyUI/Editor/data/nodes/{self.node_name}.json", "r") as f:
            self.node_dict = json.load(f)

    def to_dict(self):
        return self.node_dict

    def add_ports(self):
        y_offset = 30
        port_height = 20

        if "input" in self.node_dict:
            for i, (port_name, port_data) in enumerate(self.node_dict["input"].items()):
                input_port = InputPort(
                    text=port_name, width=100, height=port_height)
                input_port.setParentNode(self, i)
                input_port.setPosition(0, y_offset)
                self.input_ports.append(input_port)
                self.addItem(input_port)
                y_offset += port_height

        y_offset_output = 30

        if "output" in self.node_dict:
            for i, (port_name, port_data) in enumerate(self.node_dict["output"].items()):
                output_port = OutputPort(
                    text=port_name, width=100, height=port_height)
                output_port.setParentNode(self, i)
                output_port.setPosition(100, y_offset_output)
                self.output_ports.append(output_port)
                self.addItem(output_port)
                y_offset_output += port_height
        y_offset = max(y_offset, y_offset_output)
        if "parameter" in self.node_dict:
            for i, port_data in enumerate(self.node_dict["parameter"]):
                parameter_port = ParameterPort(
                    parameter=port_data["parameter"],
                    default_value=port_data["default_value"],
                    type=port_data["type"],
                    max_value=port_data.get("max_value"),
                    min_value=port_data.get("min_value"),
                    current_value=port_data.get("current_value"),
                    width=200,
                    height=port_height
                )
                parameter_port.setParentNode(self, i)
                parameter_port.setPos(0, y_offset)
                self.addItem(parameter_port)
                y_offset += port_height

        if "bool" in self.node_dict:
            for i, port_data in enumerate(self.node_dict["bool"]):
                bool_port = BoolPort(
                    parameter=port_data["parameter"],
                    default_value=port_data["default_value"],
                    current_value=port_data.get("current_value"),
                    width=200,
                    height=port_height
                )
                bool_port.setParentNode(self, i)
                bool_port.setPos(0, y_offset)
                self.addItem(bool_port)
                y_offset += port_height

        self.height = y_offset
        self.update()

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
