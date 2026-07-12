import io
import pandas as pd

csv_data = """product,price,quantity
Widget,9.99,100
Gadget,19.99,50
Gizmo,14.99,75
"""

def table_to_text(csv_text):
    df = pd.read_csv(io.StringIO(csv_text))
    rows_text = []
    for _, row in df.iterrows():
        row_desc = ", ".join(f"{col}: {val}" for col, val in row.items())
        rows_text.append(row_desc)
    return "\n".join(rows_text)

result = table_to_text(csv_data)
print(result)

assert "Widget" in result and "9.99" in result
assert result.count("\n") == 2  # 3 rows -> 2 newlines
print("OK")
