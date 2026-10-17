from typing import Literal
from pydantic import BaseModel, Field

class CreditRiskInput(BaseModel):
    Age : int = Field(ge= 18, le = 120)
    Sex : Literal["male", "female"]
    Job : int = Field(ge = 0, le = 3)
    Housing : Literal["own", "rent", "free"]
    
    Saving_account : Literal[
        "No Account",
        "little",
        "moderate",
        "quite rich",
        "rich"
    ]
    
    Checking_account : Literal[
        "No Account",
        "little",
        "moderate",
        "rich"
    ]
    
    Credit_amount : int = Field(gt = 0)
    Duration : int = Field(gt = 0)
    
    Purpose : Literal[
        "car",
        "domestic appliances",
        "education",
        "furniture/equipment",
        "radio/TV",
        "repairs",
        "vacation/others"
    ]
    
    
    