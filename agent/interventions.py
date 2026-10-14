def candidate_interventions(context, risk_category):
    """Transparent rule layer. These are candidate actions, not claimed historical actions."""
    crop = str(context.get("crop", "")).lower()
    irrigation = float(context.get("irrigation_percentage", 0) or 0)
    rainfall_anomaly = float(context.get("rainfall_anomaly", 0) or 0)
    temp_anomaly = float(context.get("temperature_anomaly", 0) or 0)

    actions = []

    if risk_category in {"High", "Severe"} or rainfall_anomaly < -0.15:
        actions += [
            "Prioritize irrigation scheduling according to local water availability.",
            "Monitor soil moisture and avoid unnecessary irrigation losses.",
            "Consider locally recommended drought/heat-tolerant varieties for the next sowing window."
        ]

    if temp_anomaly > 2:
        actions += [
            "Monitor the crop during heat-stress periods and follow locally validated heat-management practices."
        ]

    if crop in {"wheat", "maize", "rice"}:
        actions.append("Use crop- and growth-stage-specific recommendations from a local agricultural extension source.")

    if irrigation < 30:
        actions.append("Prioritize water-conservation practices because irrigation access appears limited.")

    # Deduplicate while preserving order.
    return list(dict.fromkeys(actions))
