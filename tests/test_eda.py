import pandas as pd
from tools.main import SpeedDatingEDA

def test_combine_target_and_basic_cleaning():
    df = pd.DataFrame({
        "match": [1, 0, "yes", "no", None],
        "decision": ["yes", "no", 1, 0, None],
        "age": ["21", "[22-24]", "30", None, "27"]
    })
    eda = SpeedDatingEDA(df, auto_clean=True, show_plots=False)
    # create combined target (AND semantics)
    series = eda.combine_target(["match", "decision"], new_col="target", require_all=True)
    assert "target" in eda.data.columns
    # only rows where both are truthy should be 1
    vals = eda.data["target"].astype("Int64").tolist()
    assert vals[0] == 1  # match=1, decision=yes
    assert vals[1] == 0  # match=0, decision=no
    # age parsing: '[22-24]' should become numeric midpoint (23.0)
    assert float(eda.data.loc[1, "age"]) == 23.0