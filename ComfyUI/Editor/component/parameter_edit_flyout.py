from qfluentwidgets import Flyout, FlyoutView, LineEditButton

class ParameterEditFlyoutView:
    def __init__(self, parent=None, parameter="just for test", default_value=None, type="int", max_value=None, min_value=None, current_value=None):
        super().__init__(parent)
        self.parameter = parameter
        self.default_value = default_value
        self.parameter_type = type
        self.max_value = max_value
        self.min_value = min_value
        self.current_value = current_value
        self.setupFlyoutView()

    def setupFlyoutView(self):
        self.flyout_view = FlyoutView(
            title=self.parameter, 
            content=self.tr(f"min value: {self.min_value}, max value: {self.max_value}, default value: {self.default_value}"),
            isClosable=True
        )
        line_edit = QLineEdit(self.tr("Input value"))






