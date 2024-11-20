from qfluentwidgets import MessageBoxBase, SubtitleLabel, LineEdit, CaptionLabel, PushButton
from PySide6.QtGui import QColor

class ParameterMessageBox(MessageBoxBase):
    def __init__(self, parent=None, parameter = "", default_value = None, current_value = None, type = int, max_value = None, min_value = None):
        super().__init__(parent)
        self.parameter = parameter
        self.parameter_type = type
        self.min_value = min_value
        self.max_value = max_value
        self.default_value = default_value
        self.current_value = current_value
        
        self.titleLabel = SubtitleLabel(parameter)
        self.LineEdit = LineEdit()

        self.LineEdit.setPlaceholderText(self.tr("Edit the parameter value."))
        if self.current_value is not None:
            self.LineEdit.setText(str(current_value))
        self.LineEdit.setClearButtonEnabled(True)
        
        self.pushButton = PushButton(self.tr("Reset"))
        self.pushButton.clicked.connect(self.reset_value)
        
        self.warningLabel = CaptionLabel(self.tr("Value out of range or invalid!"))
        self.warningLabel.setTextColor("#cf1010", QColor(255, 28, 32))
        self.warningLabel.hide()
        
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.LineEdit)
        self.viewLayout.addWidget(self.pushButton)
        self.viewLayout.addWidget(self.warningLabel)
        self.widget.setMinimumWidth(350)

    def validate(self):
        user_input = self.LineEdit.text()
        isValid = True
        try:
            if self.parameter_type == int:
                value = int(user_input)
            elif self.parameter_type == float:
                value = float(user_input)
            elif self.parameter_type == str:
                value = user_input   
            else:
                isValid = False  # Unsupported type
            if self.parameter_type in [int, float]:    
                if isValid and self.min_value is not None and value < self.min_value:
                    isValid = False
                if isValid and self.max_value is not None and value > self.max_value:
                    isValid = False
        except ValueError:
            # If input is not a valid integer or float, invalid
            isValid = False
    
        self.warningLabel.setHidden(isValid)
        return isValid

    def reset_value(self):
        self.LineEdit.setText(str(self.default_value))




