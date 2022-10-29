from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from classifier.config.core import config
from classifier.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    pre_processed = pre_pipeline_preparation(dataframe=input_data)
    validated_data = pre_processed[config.model_config.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleTitanicDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class AttritionDataInputSchema(BaseModel):
    satisfaction_level: Optional[int]
    last_evaluation: Optional[str]
    number_project: Optional[str]
    average_montly_hours: Optional[int]
    time_spend_company: Optional[int]
    Work_accident: Optional[int]
    promotion_last_5years: Optional[int]
    dept: Optional[float]
    salary: Optional[str]


class MultipleAttritionDataInputs(BaseModel):
    inputs: List[AttritionDataInputSchema]