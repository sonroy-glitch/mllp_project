DEFAULT_CLASS_NAMES = [
    "Badminton",
    "Cricket",
    "Karate",
    "Soccer",
    "Swimming",
    "Tennis",
    "Wrestling",
]

SPORT_INFO = {
    "Badminton": {
        "category": "Racquet Sport",
        "origin": "British India",
        "description": "Badminton is a racquet sport played with a shuttlecock over a "
        "high net, as singles or doubles on a narrow indoor court. The modern game was "
        "codified from Poona, played by British officers in nineteenth-century India.",
    },
    "Cricket": {
        "category": "Team Sport",
        "origin": "England",
        "description": "Cricket is a bat-and-ball team sport played between two sides of "
        "eleven on a grass field with a twenty-two yard central pitch. The batter, the "
        "stumps and the fielding ring set out around them are its clearest visual signature.",
    },
    "Karate": {
        "category": "Combat Sport",
        "origin": "Okinawa, Japan",
        "description": "Karate is a striking martial art contested with punches, kicks "
        "and open-hand techniques. Competitors wear a white gi with a coloured belt and "
        "compete on a matted square called a tatami.",
    },
    "Soccer": {
        "category": "Team Sport",
        "origin": "England",
        "description": "Soccer, or association football, is an eleven-a-side team sport "
        "in which players move a ball into the opposing goal without using their hands. "
        "It is played on a large rectangular grass or synthetic pitch.",
    },
    "Swimming": {
        "category": "Water Sport",
        "origin": "International",
        "description": "Competitive swimming is raced over set distances using freestyle, "
        "backstroke, breaststroke or butterfly. Lane ropes, starting blocks and the water "
        "surface itself make the venue unmistakable.",
    },
    "Tennis": {
        "category": "Racquet Sport",
        "origin": "England",
        "description": "Tennis is a racquet sport played on a rectangular court divided "
        "by a net, with a felt-covered ball. It is contested as singles or doubles on "
        "clay, grass or hard surfaces.",
    },
    "Wrestling": {
        "category": "Combat Sport",
        "origin": "International",
        "description": "Wrestling is a grappling combat sport in which opponents attempt "
        "to pin or throw one another. Competitors wear singlets and compete inside a "
        "circular contest zone marked on a padded mat.",
    },
}

FALLBACK_INFO = {
    "category": "Other",
    "origin": "International",
    "description": "A sporting activity recognised by the classifier.",
}


def describe(sport):
    return SPORT_INFO.get(sport, FALLBACK_INFO)
