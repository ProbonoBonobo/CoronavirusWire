import re
# these regex patterns help the sitemap crawler to identify local URLs that are relevant to
# coronavirus
patterns = {
    "la":
    re.compile(
        r"(\bca\b|\bl.?a.?\b|\bo.?c.?\b|\bsce\b|adelanto|anaheim|angels|arcadia|asuza|beach|bernardino|beverly|brea|burbank|california|canyon|chargers|chino|clippers|community|compton|county|covina|culver|desert|dodger|downey|downtown|ducks|fullerton|garcetti|getty|glendale|glendora|hockley|hollywood|inland.{0,1}empire|irvine|joshua.?tree|koreatown|ladwp|lakewood|lapd|lausd|lax|library|local|los.{0,1}angeles|malibu|metro|monrovia|montebello|monterey|newsom|officials|ontario|orange|palmdale|pasadena|pomona|rail|rams|rancho|regional|riverside|san.?pedro|santa.{0,1}monica|segundo|shelter|sheriff|sherman.?oaks|socal|state|stores|torrance|ucla|usc|valley|ventura|westlake|yorba)",
        re.IGNORECASE),
    "coronavirus":
    re.compile(
        r"(hospital|campus|corona|moratorium|emergency|open|resum|shut|restaurant|travel|clos|pandemic|layoff|furlough|death.?toll|recession|stimulus|medic|nurse|quarantine|shut.?down|covid|postpone|cancel|virus)",
        re.IGNORECASE)
}

# most sites simply won't respond unless you identify yourself via user-agent string
default_headers = {
    "user-agent":
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/79.0.3945.130 Safari/537.36",
    "dnt":
    "1",
    "cookie":
    "nyt-a=29482ninwfwe_efw;"
}
