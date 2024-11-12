
PERIOD = 60
MA_WINDOW_SIZE = 2 * 24 * 60# period time units
TIME_STRING = '%d.%m.%Y %H:%M:%S.000'
FILENAME_TIMESTRING = '%d.%m.%Y'
FILE_FORMAT = '(?P<pair>EURUSD)_Candlestick_1_M_(?P<direction>\\w{3})_(?P<from>\\d{2}\\.\\d{2}\\.\\d{4})-(?P<till>\\d{2}\\.\\d{2}\\.\\d{4})'
FORECAST_WINDOW = 30
FORECAST_THRESHOLD = 3e-4
PIP = 1e-4

