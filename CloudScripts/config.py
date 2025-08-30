import pandas as pd
from pandas_market_calendars import get_calendar
def trading_days_ytd(calendar_name: str = "NYSE") -> int:
    # get today’s date normalized (00:00)
    today = pd.Timestamp.today().normalize()
    start_of_year = pd.Timestamp(year=today.year, month=1, day=1)
    cal = get_calendar(calendar_name)
    # fetch the schedule between Jan 1 and today
    sched = cal.schedule(start_date=start_of_year, end_date=today)

    return len(sched)

DAYS_BACK              =  trading_days_ytd()
SP500_TICKER           = "^GSPC"
SHARES                 = None           # fixed share count or None ⇒ $‑budget
BUDGET_PER_TRADE       = 1_000          # $ when SHARES is None
SLIPPAGE_PCT           = 0.0005         # 0.05% each side
COMMISSION_PER_TRADE   = 0              # flat commission per side
MODEL_PATH             = "../model/Global/model_price_conf.keras"
TOP_N_CONF             = 100            # trades kept for dashboard & metrics



