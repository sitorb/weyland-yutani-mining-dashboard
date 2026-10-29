import pandas as pd
import numpy as np
from reportlab.lib.styles import getSampleStyleSheet
import streamlit as st
from scipy.stats import t
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)

st.title("Weyland-Yutani Mining Analytics")

CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRWWIUImVFn4C-V1KSZL1z6EYdLMfVBh45mLU-67yKAX1LDvgW1olJiu189X_eR3V-syPkosyzTitrG/pub?gid=1170474164&single=true&output=csv"

try:
    df = pd.read_csv(CSV_URL)

    st.success("Mining Data Loaded Successfully!")

    st.write("number of rows:", df.shape[0])
    st.write("number of columns:", df.shape[1])

    st.subheader("column names")
    st.write(df.columns.tolist())

    st.subheader("first five rows")
    st.write(df.head())

    st.subheader("Mining Output Statistics")

    output = df["final output"].str.replace(",", ".", regex=False).astype(float)

    st.write("Mean:", output.mean())
    st.write("Median:", output.median())
    st.write("Standard deviation:", output.std())
    st.write("Minimum output:", output.min())
    st.write("Maximum output:", output.max())

    st.subheader("Production by Mine")

    df["final output"] = (
        df["final output"]
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    def calculate_iqr(series):
        return series.quantile(0.75) - series.quantile(0.25)

    mine_stats = df.groupby("mine name")["final output"].agg(
        ["mean", "median", "std", calculate_iqr]
    )

    mine_stats = mine_stats.rename(
        columns={"calculate_iqr": "IQR"}
    )

    st.subheader("Overall Statistics — All Mines")

    overall = df["final output"]

    overall_stats = {
        "Mean": overall.mean(),
        "Median": overall.median(),
        "Standard deviation": overall.std(),
        "IQR": overall.quantile(0.75) - overall.quantile(0.25),
    }

    st.write(overall_stats)
    st.subheader("Anomaly Detection — IQR")

    q1 = df["final output"].quantile(0.25)
    q3 = df["final output"].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    anomalies = df[
        (df["final output"] < lower_bound) |
        (df["final output"] > upper_bound)
    ]

    st.write("Lower threshold:", round(lower_bound, 2))
    st.write("Upper threshold:", round(upper_bound, 2))
    st.write("Number of anomalies:", len(anomalies))
    st.dataframe(anomalies)

    import plotly.express as px

    st.subheader("Mining Production Over Time")

    chart_data = df.copy()
    chart_data["date"] = pd.to_datetime(
        chart_data["date"], format="%d.%m.%Y"
    )

    chart_type = st.selectbox(
        "Choose chart type",
        ["Line", "Bar", "Stacked bar"]
    )

    if chart_type == "Line":
        chart = px.line(
            chart_data,
            x="date",
            y="final output",
            color="mine name",
            title="Daily Production by Mine",
        )

    elif chart_type == "Bar":
        chart = px.bar(
            chart_data,
            x="date",
            y="final output",
            color="mine name",
            title="Daily Production by Mine",
            barmode="group",
        )

    else:
        chart = px.bar(
            chart_data,
            x="date",
            y="final output",
            color="mine name",
            title="Stacked Daily Production",
            barmode="stack",
        )

    degree = st.selectbox(
        "Polynomial trendline degree",
        [1, 2, 3, 4],
        index=1,
    )

    for mine_name, group in chart_data.groupby("mine name"):
        group = group.sort_values("date").copy()

        x = (
            group["date"] - chart_data["date"].min()
        ).dt.days.to_numpy()

        y = group["final output"].to_numpy()

        if len(group) <= degree:
            continue

        coefficients = np.polyfit(x, y, degree)
        trend = np.polyval(coefficients, x)

        chart.add_scatter(
            x=group["date"],
            y=trend,
            mode="lines",
            name=f"{mine_name} trend (degree {degree})",
            line={"dash": "dash"},
        )

    st.plotly_chart(chart, use_container_width=True)
    st.subheader("Anomaly Detection")

    method = st.selectbox(
    "Detection method",
    ["IQR", "Z-score", "Moving average", "Grubbs' test"]
)

    if method == "IQR":
        q1 = df["final output"].quantile(0.25)
        q3 = df["final output"].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        anomalies = df[
            (df["final output"] < lower) |
            (df["final output"] > upper)
        ]

    elif method == "Z-score":
        mean = df["final output"].mean()
        std = df["final output"].std()

        if std == 0:
            anomalies = df.iloc[0:0]
        else:
            z_scores = (df["final output"] - mean) / std
            anomalies = df[z_scores.abs() > 3]

    elif method == "Grubbs' test":
        alpha = st.selectbox(
            "Significance level",
            [0.01, 0.05, 0.10],
            index=1,
        )

        anomaly_parts = []

        for mine_name, group in df.groupby("mine name"):
            values = group["final output"]
            n = len(values)

            if n < 3 or values.std() == 0:
                continue

            mean = values.mean()
            std = values.std()
            deviations = (values - mean).abs()
            index = deviations.idxmax()
            g_stat = deviations.loc[index] / std

            t_critical = t.ppf(
                1 - alpha / (2 * n), n - 2
            )

            g_critical = (
                (n - 1) / np.sqrt(n)
                * np.sqrt(t_critical**2 / (n - 2 + t_critical**2))
            )

            if g_stat > g_critical:
                anomaly_parts.append(df.loc[[index]])

        anomalies = (
            pd.concat(anomaly_parts)
            if anomaly_parts
            else df.iloc[0:0]
        )

    else:
        chart_data = df.copy()
        chart_data["date"] = pd.to_datetime(
            chart_data["date"], format="%d.%m.%Y"
        )
        chart_data = chart_data.sort_values(["mine name", "date"])

        chart_data["moving average"] = (
            chart_data.groupby("mine name")["final output"]
            .transform(lambda s: s.rolling(7, min_periods=3).mean())
        )

        chart_data["distance (%)"] = (
            (chart_data["final output"] - chart_data["moving average"])
            / chart_data["moving average"].abs().replace(0, float("nan"))
            * 100
        )

        threshold = st.slider(
            "Moving-average deviation threshold (%)",
            min_value=10,
            max_value=100,
            value=30,
            step=5,
        )

        anomalies = chart_data[
            chart_data["distance (%)"].abs() > threshold
        ]

    st.write("Number of anomalies:", len(anomalies))
    st.dataframe(anomalies)

    st.subheader("PDF Report")

    if st.button("Generate PDF report"):
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        story = []

        story.append(Paragraph("Mining Analytics Report", getSampleStyleSheet()["Title"]))        
        story.append(Spacer(1, 12))

        report_stats = df.groupby("mine name")["final output"].agg(
            ["mean", "median", "std"]
        ).reset_index()

        report_stats["IQR"] = df.groupby("mine name")["final output"].apply(
            lambda s: s.quantile(0.75) - s.quantile(0.25)
        ).values

        table_data = [list(report_stats.columns)]
        table_data += report_stats.round(2).astype(str).values.tolist()

        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))

        story.append(Paragraph("Statistics by Mine",
                               __import__("reportlab").lib.styles.getSampleStyleSheet()["Heading2"]))
        story.append(table)
        story.append(Spacer(1, 16))
        story.append(Paragraph(
            f"Selected anomaly method: {method}",
            __import__("reportlab").lib.styles.getSampleStyleSheet()["Normal"]
        ))
        story.append(Paragraph(
            f"Detected anomaly rows: {len(anomalies)}",
            __import__("reportlab").lib.styles.getSampleStyleSheet()["Normal"]
        ))

        chart_image = BytesIO()
        chart.write_image(chart_image, format="png")
        chart_image.seek(0)
        story.append(Spacer(1, 12))
        story.append(Image(chart_image, width=450, height=260))

        doc.build(story)
        pdf_buffer.seek(0)

        st.download_button(
            "Download PDF report",
            data=pdf_buffer.getvalue(),
            file_name="mining_analytics_report.pdf",
            mime="application/pdf",
        )
except Exception as e:
    st.error(f"Error loading data: {e}")