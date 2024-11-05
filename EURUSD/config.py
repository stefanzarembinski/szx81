CONFIG = {
    'period': 60,
    'ma_window_size': 2 * 24 * 60, # period time units
    'timestring': '%d.%m.%Y %H:%M:%S.000',
    'filename_timestring': '%d.%m.%Y',
    'file_format': '(?P<pair>EURUSD)_Candlestick_1_M_(?P<direction>\\w{3})_(?P<from>\\d{2}\\.\\d{2}\\.\\d{4})-(?P<till>\\d{2}\\.\\d{2}\\.\\d{4})',
    'forecast_window': 30,
    'forecast_threshold': 1e-4,
}
