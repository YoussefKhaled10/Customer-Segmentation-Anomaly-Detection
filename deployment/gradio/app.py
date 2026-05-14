import gradio as gr
import requests

API_URL = "http://localhost:5000/predict"


def format_api_error(error_detail):
    """
    Convert FastAPI validation errors into user-friendly messages.
    """
    # Case 1: FastAPI validation error list
    if isinstance(error_detail, list):
        messages = []

        for err in error_detail:
            field = err.get("loc", ["input"])[-1]
            msg = err.get("msg", "Invalid value")
            value = err.get("input", None)

            # Custom Arabic/clear messages
            if field == "recency":
                field_name = "Recency"
                explanation = "Number of days since last purchase must be a number greater than or equal to 0"
            elif field == "frequency":
                field_name = "Frequency"
                explanation = "Number of purchases must be a number greater than or equal to 0"
            elif field == "monetary":
                field_name = "Monetary"
                explanation = "Total customer spend must be a number greater than or equal to 0"
            else:
                field_name = str(field)
                explanation = msg

            messages.append(
                f"Error in {field_name}\n"
                f"Entered value: {value}\n"
                f"Reason: {explanation}"
            )

        return "\n\n".join(messages)
    # Case 2: Business validation error string
    if isinstance(error_detail, str):
        return f"{error_detail}"

    # Case 3: Unknown error format
    return f"Unexpected error: {error_detail}"


def predict_customer_ui(recency, frequency, monetary):
    payload = {
        "recency": recency,
        "frequency": frequency,
        "monetary": monetary
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)

        # Handle API validation/business errors
        if response.status_code != 200:
            try:
                error_detail = response.json().get("detail", "Unknown API error")
            except Exception:
                error_detail = response.text

            clean_error = format_api_error(error_detail)

            return (
                "Invalid Input",
                "Invalid Input",
                "Invalid Input",
                "Invalid Input",
                clean_error
            )

        result = response.json()

        status = result.get("status", "unknown")
        segment_id = result.get("segment_id", "N/A")
        segment = result.get("segment", "Unknown")
        avg_order_value = result.get("avg_order_value", "N/A")
        anomaly = result.get("anomaly", False)
        action = result.get("recommended_action", "N/A")
        message = result.get("message", "")

        anomaly_text = "Yes ⚠️" if anomaly else "No"

        return (
            str(segment_id),
            segment,
            str(avg_order_value),
            anomaly_text,
            f"{action}\n\nStatus: {status}\nMessage: {message}"
        )

    except requests.exceptions.ConnectionError:
        return (
            "Error",
            "Error",
            "Error",
            "Error",
            "FastAPI server is not running. Please start it on port 5000."
        )

    except requests.exceptions.Timeout:
        return (
            "Error",
            "Error",
            "Error",
            "Error",
            "Request timed out. The API took too long to respond."
        )

    except Exception as e:
        return (
            "Error",
            "Error",
            "Error",
            "Error",
            str(e)
        )


with gr.Blocks(title="Customer Segmentation & Anomaly Detection") as demo:

    gr.Markdown(
        """
        # Customer Segmentation & Anomaly Detection

        This app predicts the customer segment using **KMeans**
        and detects unusual customer behavior using **Isolation Forest**.

        Enter customer behavior metrics below:
        """
    )

    with gr.Row():
        recency = gr.Number(
            label="Recency",
            value=30,
            info="Days since last purchase. Must be >= 0"
        )

        frequency = gr.Number(
            label="Frequency",
            value=2,
            info="Number of purchases. Must be >= 0"
        )

        monetary = gr.Number(
            label="Monetary",
            value=500,
            info="Total customer spend. Must be >= 0"
        )

    predict_btn = gr.Button("Predict Customer Segment", variant="primary")

    gr.Markdown("## Prediction Result")

    with gr.Row():
        segment_id_out = gr.Textbox(label="Segment ID")
        segment_out = gr.Textbox(label="Customer Segment")

    with gr.Row():
        avg_order_value_out = gr.Textbox(label="Avg Order Value")
        anomaly_out = gr.Textbox(label="Anomaly Status")

    action_out = gr.Textbox(
        label="Recommended Action / Message",
        lines=6
    )

    predict_btn.click(
        fn=predict_customer_ui,
        inputs=[recency, frequency, monetary],
        outputs=[
            segment_id_out,
            segment_out,
            avg_order_value_out,
            anomaly_out,
            action_out
        ]
    )

demo.launch()