# Power BI forecast chart: setup (about 10 minutes)

Files here: `forecast_powerbi.csv` (main table), `metrics.csv`, `measures.dax`, `powerbi_preview.png` (target look).
`forecast_powerbi.csv` is your `outputs/forecast_results.csv` plus a `period` column (History / Hold-out / Forecast)
and one bridge point (Dec 2017) so the red forecast line joins the actual line.

1. Power BI Desktop > Home > Get data > Text/CSV > load `forecast_powerbi.csv`. Load again for `metrics.csv`.
2. Check `month` is type Date (Transform data > column header icon).
3. Insert a **Line chart**.
   - X-axis: `month` (click the arrow and pick *month*, not Date Hierarchy)
   - Y-axis: `actual`, `seasonal_naive`, `sarima`, `gradient_boosting`, `future_forecast` (set each to Sum or Don't summarize)
   - Format: Lines > make `future_forecast` thick red, the three model lines dashed. Title: "Superstore Monthly Sales: Actual vs Forecast".
4. Insert 3 **Card** visuals using the measures in `measures.dax` (Best Model, Best MAPE %, Best RMSE).
5. Insert a **Table** visual with the `metrics` columns.
6. File > Save as `superstore_forecast.pbix`.
