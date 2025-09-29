from .market_data import MarketDataService
import threading as th

class CalibrationService:
    def __init__(self):
        # Initialize MarketDataService
        self.market_data_service = MarketDataService()
        self.implied_volatilities = {}
        self.volatilities_serfaces = {}
        self.model_params = {}
        self.forward_curves = {}
        self.interest_rates = {}
        self.dividend_yields = {}
        self.calibration_status = {}
        self.calibration_weights = {}
        self.last_calibration = {}
        self.parameter_history = {}
        self.calibration_lock = th.Lock()
