from enum import Enum


class ActivityType(Enum):
    """
        Provides a comprehensive list of the activities.

        Accessing the data:
            * .value       -> Returns the integer ID (e.g., 1)
            * .name        -> Returns the exact name of the enum member (e.g., 'SITTING')
            * .description -> Returns the detailed explanation of the activity
    """
    TRANSITION = (
        0,
        "Transition period to arrive at the starting location of the next activity."
    )

    SITTING = (
        1,
        "Sitting still while reading."
    )

    ASCENDING_DESCENDING_STAIRS = (
        2,
        "Climbing six floors up and going down again, repeating this twice. "
        "Note: for subjects S1 and S2, going down was performed only once."
    )

    TABLE_SOCCER = (
        3,
        "Playing table soccer, 1 vs. 1 with the supervisor of the data collection."
    )

    CYCLING = (
        4,
        "Performed outdoors following a defined route of about 2km length "
        "with varying road conditions (gravel, paved)."
    )

    CAR_DRIVING = (
        5,
        "Subjects followed a defined route which took about 15 minutes to complete. "
        "The route included driving on different streets in a small city"
        " as well as driving on country roads."
    )

    LUNCH_BREAK = (
        6,
        "The activity included queuing and fetching food, eating, and talking at the table."
    )

    WALKING = (
        7,
        "This activity was carried within a specified route with some detour."
    )

    WORKING = (
        8,
        "Subjects returned to their desk and worked as if not participating in this "
        "study. For each subject, work mainly consisted of working on a computer."
    )

    def __new__(cls, value, description):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.description = description
        return obj
