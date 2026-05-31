"""
Pydantic output schema for youth program extraction (aligned with extraction rules).
"""

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class AgeGroup(BaseModel):
    minAge: str = Field(description="Minimum age in years as number string, or 'no data available'")
    maxAge: str = Field(description="Maximum age in years as number string, or 'no data available'")


class ActivityRecurringBlock(BaseModel):
    days: List[str] = Field(default_factory=list)
    activityRecurring: bool = False


class TimeBlock(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: str = Field(default="", alias="from")
    to: str = ""


class ScheduleEntry(BaseModel):
    day: str = ""
    startTime: str = ""
    endTime: str = ""
    frequency: str = ""


class PriceEntry(BaseModel):
    priceUnit: str = ""
    priceType: str = ""
    pricePerParticipant: str = ""
    pricePerHour: str = ""
    classDuration: str = ""
    halfOrFullDay: str = ""
    duration: str = ""
    title: str = ""
    noOfHours: str = ""
    noOfDays: str = ""
    noOfWeeks: str = ""


class Program(BaseModel):
    name: str = ""
    description: str = ""
    ageGroup: AgeGroup = Field(default_factory=lambda: AgeGroup(minAge="", maxAge=""))
    activityRecurring: ActivityRecurringBlock = Field(
        default_factory=lambda: ActivityRecurringBlock(days=[], activityRecurring=False)
    )
    time: TimeBlock = Field(default_factory=TimeBlock)
    schedule: ScheduleEntry = Field(default_factory=ScheduleEntry)
    schedules: List[ScheduleEntry] = Field(default_factory=list)
    offerDiscount: str = ""
    maxNumberOfStudents: str = ""
    parentalSupervisionRequired: str = ""
    indoorOroutdoor: str = ""
    inpersonOrVirtual: str = ""
    joiningLink: str = ""
    type: str = ""
    pricingData: str = ""
    isFreeTrial: bool = False
    prices: List[PriceEntry] = Field(default_factory=list)


class ProviderProfile(BaseModel):
    """Provider-level details extracted from homepage + about/contact pages.

    Added alongside the existing program schema (additive only — no changes to
    Program / AgeGroup / PriceEntry / ScheduleEntry / TimeBlock).
    """

    name: str = ""
    address: str = ""
    phone: str = ""
    email: str = ""
    description: str = ""
    categories: List[str] = Field(default_factory=list)
    subjects: List[str] = Field(default_factory=list)


class ProgramsResponse(BaseModel):
    programs: List[Program] = Field(default_factory=list)
    provider: ProviderProfile = Field(default_factory=ProviderProfile)
