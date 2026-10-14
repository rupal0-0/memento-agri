import os


class MementoLLM:

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")

        self.model = os.getenv(
            "OPENAI_MODEL",
            "gpt-4.1-mini"
        )


    # --------------------------------------------------
    # Safe value helpers
    # --------------------------------------------------

    @staticmethod
    def _num(data, key, default=0.0):

        value = data.get(key, default)

        if value is None:
            return default

        try:
            return float(value)
        except (TypeError, ValueError):
            return default


    @staticmethod
    def _text(data, key, default="Unknown"):

        value = data.get(key, default)

        if value is None:
            return default

        return str(value)


    # --------------------------------------------------
    # Main generation function
    # --------------------------------------------------

    def generate(
        self,
        query,
        context,
        risk,
        execution
    ):

        # If an OpenAI API key exists, use the real LLM.
        # Otherwise use the local deterministic research
        # prototype response.

        if self.api_key:

            try:
                return self._openai_generate(
                    query,
                    context,
                    risk,
                    execution
                )

            except Exception as e:

                print(
                    "LLM API failed. "
                    "Using research prototype response.",
                    e
                )

        return self._mock(
            query,
            context,
            risk,
            execution
        )


    # --------------------------------------------------
    # OpenAI generation
    # --------------------------------------------------

    def _openai_generate(
        self,
        query,
        context,
        risk,
        execution
    ):

        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key
        )

        state = self._text(
            context,
            "state"
        )

        district = self._text(
            context,
            "district"
        )

        crop = self._text(
            context,
            "crop"
        )

        rainfall = self._num(
            context,
            "rainfall"
        )

        temperature = self._num(
            context,
            "temperature"
        )

        humidity = self._num(
            context,
            "humidity"
        )

        rainfall_anomaly = self._num(
            context,
            "rainfall_anomaly"
        )

        temperature_anomaly = self._num(
            context,
            "temperature_anomaly"
        )

        climate_risk = self._num(
            context,
            "climate_risk"
        )

        prompt = f"""
You are Memento-Agri, an agricultural climate advisory
agent.

Your LLM weights are frozen.

You must use the retrieved historical experiences
provided by the Memento memory system.

Farmer context:

State: {state}
District: {district}
Crop: {crop}

Current climate:

Rainfall: {rainfall:.2f} mm
Temperature: {temperature:.2f} °C
Humidity: {humidity:.2f} %

Rainfall anomaly:
{rainfall_anomaly:.2%}

Temperature anomaly:
{temperature_anomaly:.2%}

Estimated climate risk:
{climate_risk:.2%}

Farmer question:

{query}

Retrieved historical experiences:

{context.get("retrieved_cases", [])}

Planner output:

{execution}

Provide:

1. Current climate-risk interpretation
2. Main factors causing the risk
3. Practical actions the farmer can consider
4. Crop-specific advice
5. What to monitor
6. Important uncertainty/caution

Do not claim that a historical observation proves
that an intervention caused a particular yield.

Keep the advice practical and understandable.
"""

        response = client.responses.create(
            model=self.model,
            input=prompt
        )

        return response.output_text


    # --------------------------------------------------
    # Deterministic research prototype
    # --------------------------------------------------

    def _mock(
        self,
        query,
        context,
        risk,
        execution
    ):

        state = self._text(
            context,
            "state"
        )

        district = self._text(
            context,
            "district"
        )

        crop = self._text(
            context,
            "crop"
        )

        rainfall = self._num(
            context,
            "rainfall"
        )

        temperature = self._num(
            context,
            "temperature"
        )

        humidity = self._num(
            context,
            "humidity"
        )

        rainfall_anomaly = self._num(
            context,
            "rainfall_anomaly"
        )

        temperature_anomaly = self._num(
            context,
            "temperature_anomaly"
        )

        humidity_anomaly = self._num(
            context,
            "humidity_anomaly"
        )

        wind_anomaly = self._num(
            context,
            "wind_anomaly"
        )

        solar_anomaly = self._num(
            context,
            "solar_anomaly"
        )

        climate_risk = self._num(
            context,
            "climate_risk"
        )


        # --------------------------------------------------
        # Risk category
        # --------------------------------------------------

        if climate_risk < 0.25:

            risk_label = "Low"

        elif climate_risk < 0.50:

            risk_label = "Moderate"

        elif climate_risk < 0.75:

            risk_label = "High"

        else:

            risk_label = "Severe"


        # --------------------------------------------------
        # Determine dominant climate stress
        # --------------------------------------------------

        stresses = {

            "rainfall": abs(
                rainfall_anomaly
            ),

            "temperature": abs(
                temperature_anomaly
            ),

            "humidity": abs(
                humidity_anomaly
            ),

            "wind": abs(
                wind_anomaly
            ),

            "solar radiation": abs(
                solar_anomaly
            )
        }

        dominant = max(
            stresses,
            key=stresses.get
        )


        # --------------------------------------------------
        # Generate recommendations
        # --------------------------------------------------

        recommendations = []


        if rainfall_anomaly < -0.5:

            recommendations.append(
                "Monitor soil moisture closely and "
                "prioritize efficient irrigation if "
                "irrigation is available."
            )

            recommendations.append(
                "Consider moisture-conserving practices "
                "such as mulching and reducing unnecessary "
                "soil evaporation."
            )


        elif rainfall_anomaly > 0.5:

            recommendations.append(
                "Monitor fields for waterlogging and "
                "ensure drainage channels remain clear."
            )

            recommendations.append(
                "Avoid unnecessary irrigation while "
                "soil moisture remains high."
            )


        if temperature_anomaly > 0.5:

            recommendations.append(
                "Monitor crop heat stress, especially "
                "during the hottest part of the day."
            )

            recommendations.append(
                "Maintain adequate soil moisture where "
                "water availability permits."
            )


        if humidity_anomaly > 0.75:

            recommendations.append(
                "High humidity can increase disease risk. "
                "Inspect crops regularly for fungal or "
                "moisture-related symptoms."
            )


        if wind_anomaly > 0.75:

            recommendations.append(
                "Monitor exposed crops for lodging or "
                "physical wind damage."
            )


        if solar_anomaly < -0.75:

            recommendations.append(
                "Monitor crop development because reduced "
                "solar radiation may affect photosynthetic "
                "activity."
            )


        if not recommendations:

            recommendations.append(
                "Continue regular monitoring of soil "
                "moisture, crop condition and local weather."
            )


        # --------------------------------------------------
        # Retrieved memory
        # --------------------------------------------------

        retrieved = context.get(
            "retrieved_cases",
            []
        )

        if retrieved is None:

            retrieved = []


        memory_statement = (

            f"Memento memory retrieved "
            f"{len(retrieved)} historical "
            f"similar case(s) for comparison."
        )


        # --------------------------------------------------
        # Final advisory
        # --------------------------------------------------

        advisory = []

        advisory.append(
            f"### Climate Risk: {risk_label} "
            f"({climate_risk:.1%})"
        )

        advisory.append("")

        advisory.append(
            f"**Location:** {district}, {state}"
        )

        advisory.append(
            f"**Crop:** {crop}"
        )

        advisory.append("")

        advisory.append(
            "### Current Conditions"
        )

        advisory.append(
            f"- Rainfall: {rainfall:.1f} mm"
        )

        advisory.append(
            f"- Rainfall anomaly: "
            f"{rainfall_anomaly:.1%}"
        )

        advisory.append(
            f"- Temperature: "
            f"{temperature:.1f} °C"
        )

        advisory.append(
            f"- Temperature anomaly: "
            f"{temperature_anomaly:.1%}"
        )

        advisory.append(
            f"- Humidity: "
            f"{humidity:.1f}%"
        )

        advisory.append("")

        advisory.append(
            f"### Main Climate Signal"
        )

        advisory.append(
            f"The strongest detected climate deviation "
            f"is associated with **{dominant}**."
        )

        advisory.append("")

        advisory.append(
            "### Recommended Actions"
        )

        for recommendation in recommendations:

            advisory.append(
                f"- {recommendation}"
            )

        advisory.append("")

        advisory.append(
            "### Memento Memory"
        )

        advisory.append(
            memory_statement
        )

        advisory.append(
            "Historical cases are used as contextual "
            "experience. They should not be interpreted "
            "as proof that a specific intervention "
            "caused a particular yield."
        )

        advisory.append("")

        advisory.append(
            "### Important Note"
        )

        advisory.append(
            "This is a research prototype and should "
            "supplement, not replace, local agricultural "
            "extension advice and verified weather "
            "information."
        )

        return "\n".join(
            advisory
        )


# ============================================================
# Backward-compatible wrapper
# ============================================================

class LLMClient(MementoLLM):
    """
    Backward-compatible LLM client.

    agent/agent.py currently imports:
        from agent.llm import LLMClient

    This wrapper keeps that interface while using
    the new MementoLLM implementation.
    """

    pass
