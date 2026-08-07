from __future__ import annotations

import hashlib
import re
import sqlite3
import struct
from functools import lru_cache
from pathlib import Path
from typing import Any

from leaf_codec import (
    BRANCH_PROTOCOL_VERSION,
    LEXICAL_PAYLOAD_VERSION,
    MACRO_GUARD_END,
    MACRO_GUARD_START,
    _bits_to_bytes,
    _bits_to_macro_symbols,
    _bytes_to_bits,
    _crc16,
    _macro_symbols_to_bits,
)


WORD_PATTERN = re.compile(r"^[A-Z]+(?:['-][A-Z]+)*$")
TOKEN_PATTERN = re.compile(r"[A-Z]+(?:['-][A-Z]+)*|\s+|.", re.DOTALL)
PUNCTUATION = (".", ",", "?", "!", ";", ":", "—", "–", "(", ")", '"', "/", "&", "%", "'", "-")
PUNCTUATION_INDEX = {symbol: index for index, symbol in enumerate(PUNCTUATION)}
BRANCH_FIXED_FORMAT = ">2sB4s2sHHB"
COMPACT_GUARD_START = [0, 3, 2, 1]
COMPACT_GUARD_END = list(reversed(COMPACT_GUARD_START))
COMMON_WORDS = """
THE OF AND TO A IN IS IT YOU THAT HE WAS FOR ON ARE AS WITH HIS THEY I AT BE
THIS HAVE FROM OR ONE HAD BY WORD BUT NOT WHAT ALL WERE WE WHEN YOUR CAN SAID
THERE USE AN EACH WHICH SHE DO HOW THEIR IF WILL UP OTHER ABOUT OUT MANY THEN
THEM THESE SO SOME HER WOULD MAKE LIKE HIM INTO TIME HAS LOOK TWO MORE WRITE GO
SEE NUMBER NO WAY COULD PEOPLE MY THAN FIRST WATER BEEN CALL WHO OIL ITS NOW
FIND LONG DOWN DAY DID GET COME MADE MAY PART OVER NEW SOUND TAKE ONLY LITTLE
WORK KNOW PLACE YEAR LIVE ME BACK GIVE MOST VERY AFTER THING OUR JUST NAME GOOD
SENTENCE MAN THINK SAY GREAT WHERE HELP THROUGH MUCH BEFORE LINE RIGHT TOO MEAN
OLD ANY SAME TELL BOY FOLLOW CAME WANT SHOW ALSO AROUND FORM THREE SMALL SET PUT
END DOES ANOTHER WELL LARGE MUST BIG EVEN SUCH BECAUSE TURN HERE WHY ASK WENT
MEN READ NEED LAND DIFFERENT HOME US MOVE TRY KIND HAND PICTURE AGAIN CHANGE
OFF PLAY SPELL AIR AWAY ANIMAL HOUSE POINT PAGE LETTER MOTHER ANSWER FOUND STUDY
STILL LEARN SHOULD WORLD HIGH EVERY NEAR ADD FOOD BETWEEN OWN BELOW COUNTRY
PLANT LAST SCHOOL FATHER KEEP TREE NEVER START CITY EARTH EYE LIGHT THOUGHT
HEAD UNDER STORY SAW LEFT DON'T FEW WHILE ALONG MIGHT CLOSE SOMETHING SEEM NEXT
HARD OPEN EXAMPLE BEGIN LIFE ALWAYS THOSE BOTH PAPER TOGETHER GOT GROUP OFTEN
RUN IMPORTANT UNTIL CHILDREN SIDE FEET CAR MILE NIGHT WALK WHITE SEA BEGAN GROW
TOOK RIVER FOUR CARRY STATE ONCE BOOK HEAR STOP WITHOUT SECOND LATER MISS IDEA
ENOUGH EAT FACE WATCH FAR INDIAN REAL ALMOST LET ABOVE GIRL SOMETIMES MOUNTAIN
CUT YOUNG TALK SOON LIST SONG BEING LEAVE FAMILY BODY MUSIC COLOR STAND SUN
QUESTIONS FISH AREA MARK DOG HORSE BIRDS PROBLEM COMPLETE ROOM KNEW SINCE EVER
PIECE TOLD USUALLY DIDN'T FRIEND EASY HEARD ORDER RED DOOR SURE BECOME TOP SHIP
ACROSS TODAY DURING SHORT BETTER BEST HOWEVER LOW HOURS BLACK PRODUCTS HAPPENED
WHOLE MEASURE REMEMBER EARLY WAVES REACHED LISTEN WIND ROCK SPACE COVER FAST
SEVERAL HOLD HIMSELF TOWARD FIVE STEP MORNING PASSED VOWEL TRUE HUNDRED AGAINST
PATTERN NUMERAL TABLE NORTH SLOWLY MONEY MAP FARM PULLED DRAW VOICE SEEN COLD
CRIED PLAN NOTICE SOUTH SING WAR GROUND FALL KING TOWN I'LL UNIT FIGURE CERTAIN
FIELD TRAVEL WOOD FIRE UPON DONE ENGLISH ROAD HALF TEN FLY GAVE BOX FINALLY WAIT
CORRECT OH QUICKLY PERSON BECAME SHOWN MINUTES STRONG VERB STARS FRONT FEEL FACT
INCH STREET DECIDED CONTAIN COURSE SURFACE PRODUCE BUILD OCEAN CLASS NOTE NOTHING
REST CAREFULLY SCIENTISTS INSIDE WHEELS STAY GREEN KNOWN ISLAND WEEK LESS MACHINE
BASE AGO STOOD PLANE SYSTEM BEHIND RAN ROUND BOAT GAME FORCE BROUGHT UNDERSTAND
WARM COMMON BRING EXPLAIN DRY THOUGH LANGUAGE SHAPE DEEP THOUSANDS YES CLEAR
EQUATION YET GOVERNMENT FILLED HEAT FULL HOT CHECK OBJECT AM RULE AMONG NOUN
POWER CANNOT ABLE SIX SIZE DARK BALL MATERIAL SPECIAL HEAVY FINE PAIR CIRCLE
INCLUDE BUILT MATTER SQUARE SYLLABLE PERHAPS BILL FELT SUDDENLY TEST DIRECTION
CENTER FARMERS READY ANYTHING DIVIDED GENERAL ENERGY SUBJECT EUROPE MOON REGION
RETURN BELIEVE DANCE MEMBERS PICK SIMPLE CELLS PAINT MIND LOVE CAUSE RAIN EXERCISE
EGGS TRAIN BLUE WISH DROP DEVELOPED WINDOW DIFFERENCE DISTANCE HEART SITE SUMMER
WALL FOREST PROBABLY LEGS SAT MAIN WINTER WIDE WRITTEN LENGTH REASON KEPT INTEREST
ARMS BROTHER RACE PRESENT BEAUTIFUL STORE JOB EDGE PAST SIGN RECORD FINISHED
DISCOVERED WILD HAPPY BESIDE GONE SKY GLASS MILLION WEST LAY WEATHER ROOT
INSTRUMENTS MEET THIRD MONTH PARAGRAPH RAISED REPRESENT SOFT WHETHER CLOTHES
FLOWERS SHALL TEACHER HELD DESCRIBE DRIVE CROSS SPEAK SOLVE APPEAR METAL SON
EITHER ICE SLEEP VILLAGE FACTORS RESULT JUMP SNOW RIDE CARE FLOOR HILL PUSH BABY
BUY CENTURY OUTSIDE EVERYTHING TALL ALREADY INSTEAD PHRASE SOIL BED COPY FREE
HOPE SPRING CASE LAUGHED NATION QUITE TYPE THEMSELVES TEMPERATURE BRIGHT LEAD
EVERYONE METHOD SECTION LAKE IRON WITHIN DICTIONARY HAIR AGE AMOUNT SCALE POUNDS
ALTHOUGH PER BROKEN MOMENT TINY POSSIBLE GOLD MILK QUIET NATURAL LOT STONE ACT
BUILD MIDDLE SPEED COUNT CONSONANT SOMEONE SAIL ROLLED BEAR WONDER SMILED ANGLE
FRACTION AFRICA KILLED MELODY BOTTOM TRIP HOLE POOR LET'S FIGHT SURPRISE FRENCH
DIED BEAT EXACTLY REMAIN DRESS CAT COULDN'T FINGERS ROW LEAST CATCH CLIMBED WROTE
SHOUTED CONTINUED ITSELF ELSE PLAINS GAS ENGLAND BURNING DESIGN JOINED FOOT LAW
EARS GRASS YOU'RE GREW SKIN VALLEY CENTS KEY PRESIDENT BROWN TROUBLE COOL CLOUD
LOST SENT SYMBOL WEAR BAD SAVE EXPERIMENT ENGINE ALONE DRAWING EAST CHOOSE SINGLE
TOUCH INFORMATION EXPRESS MOUTH YARD EQUAL DECIMAL YOURSELF CONTROL PRACTICE
REPORT STRAIGHT RISE STATEMENT STICK PARTY SEEDS SUPPOSE WOMAN COAST BANK PERIOD
WIRE PAY CLEAN VISIT BIT WHOSE RECEIVED GARDEN PLEASE STRANGE CAUGHT FELL TEAM
GOD CAPTAIN DIRECT RING SERVE CHILD DESERT INCREASE HISTORY COST MAYBE BUSINESS
SEPARATE BREAK UNCLE HUNT FLOW LADY STUDENTS HUMAN ART FEELING SUPPLY CORNER
ELECTRIC INSECTS CROPS TONE HIT SAND DOCTOR PROVIDE THUS WON'T COOK BONES TAIL
BOARD MODERN COMPOUND MINE WASN'T FIT ADDITION BELONG SAFE SOLDIERS GUESS SILENT
TRADE RATHER COMPARE CROWD POEM ENJOY ELEMENTS INDICATE EXCEPT EXPECT FLAT SEVEN
INTERESTING SENSE STRING BLOW FAMOUS VALUE WINGS MOVEMENT POLE EXCITING BRANCHES
THICK BLOOD LIE SPOT BELL FUN LOUD CONSIDER SUGGESTED THIN POSITION ENTERED FRUIT
TIED RICH DOLLARS SEND SIGHT CHIEF JAPANESE STREAM PLANETS RHYTHM SCIENCE MAJOR
OBSERVE TUBE NECESSARY WEIGHT MEAT LIFT PROCESS ARMY HAT PROPERTY PARTICULAR
SWIM TERMS CURRENT PARK SELL SHOULDER INDUSTRY WASH BLOCK SPREAD CATTLE WIFE
SHARP COMPANY RADIO WE'LL ACTION CAPITAL FACTORIES SETTLED YELLOW ISN'T SOUTHERN
TRUCK FAIR PRINT WOULDN'T AHEAD CHANCE BORN LEVEL TRIANGLE MOLECULE FRANCE
REPEATED COLUMN WESTERN CHURCH SISTER OXYGEN PLURAL VARIOUS AGREED OPPOSITE WRONG
CHART PREPARED PRETTY SOLUTION FRESH SHOP SUFFIX ESPECIALLY SHOES ACTUALLY NOSE
AFRAID DEAD SUGAR ADJECTIVE FIG OFFICE HUGE GUN SIMILAR DEATH SCORE FORWARD
STRETCHED EXPERIENCE ROSE ALLOW FEAR WORKERS WASHINGTON GREEK WOMEN BOUGHT LED
MARCH NORTHERN CREATE BRITISH DIFFICULT MATCH WIN STEEL TOTAL DEAL DETERMINE
EVENING NOR ROPE COTTON APPLE DETAILS ENTIRE CORN SUBSTANCES SMELL TOOLS
CONDITIONS COWS TRACK ARRIVED LOCATED SIR SEAT DIVISION EFFECT UNDERLINE VIEW
PROJECT IMAGES MODEL REALITY DIGITAL MEDIA ARCHIVE TEXT DATA LEAF VEIN PROTOCOL
""".split()


@lru_cache(maxsize=4)
def lexical_codebook(database_path: str) -> tuple[tuple[str, ...], str]:
    path = Path(database_path)
    if not path.is_file():
        raise FileNotFoundError(f"Lexical codebook database not found: {path}")
    words: list[str] = []
    seen: set[str] = set()
    for word in COMMON_WORDS:
        if word not in seen and WORD_PATTERN.fullmatch(word):
            seen.add(word)
            words.append(word)
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT lemma FROM lexeme WHERE zipf_frequency IS NOT NULL "
            "ORDER BY zipf_frequency DESC, lemma ASC"
        )
        for (lemma,) in rows:
            word = str(lemma).upper()
            if word in seen or not WORD_PATTERN.fullmatch(word):
                continue
            seen.add(word)
            words.append(word)
            if len(words) == 16384:
                break
    if len(words) < 4096:
        raise ValueError("The lexical database does not contain enough codec entries.")
    fingerprint = hashlib.sha256("\n".join(words).encode("utf-8")).hexdigest()[:16]
    return tuple(words), fingerprint


def _append_int(bits: list[int], value: int, width: int) -> None:
    bits.extend((value >> shift) & 1 for shift in range(width - 1, -1, -1))


def _read_int(bits: list[int], offset: int, width: int) -> tuple[int, int]:
    if offset + width > len(bits):
        raise ValueError("Incomplete lexical token.")
    value = 0
    for bit in bits[offset : offset + width]:
        value = (value << 1) | bit
    return value, offset + width


def encode_lexical_bits(text: str, database_path: Path) -> dict[str, Any]:
    words, fingerprint = lexical_codebook(str(database_path.resolve()))
    word_ids = {word: index for index, word in enumerate(words)}
    bits: list[int] = []
    token_count = 0
    dictionary_tokens = 0
    raw_tokens = 0
    for token in TOKEN_PATTERN.findall(text):
        token_count += 1
        word_id = word_ids.get(token) if WORD_PATTERN.fullmatch(token) else None
        if word_id is not None and word_id < 256:
            bits.extend((0, 0))
            _append_int(bits, word_id, 8)
            dictionary_tokens += 1
        elif word_id is not None and word_id < 4096:
            bits.extend((0, 1))
            _append_int(bits, word_id, 12)
            dictionary_tokens += 1
        elif word_id is not None and word_id < 16384:
            bits.extend((1, 0, 0))
            _append_int(bits, word_id, 14)
            dictionary_tokens += 1
        elif WORD_PATTERN.fullmatch(token) and len(token) <= 63 and token.isascii():
            bits.extend((1, 0, 1))
            _append_int(bits, len(token), 6)
            for character in token:
                _append_int(bits, ord(character), 7)
            raw_tokens += 1
        elif token == " ":
            bits.extend((1, 1, 0, 0))
        elif token == "\n\n":
            bits.extend((1, 1, 0, 1))
        elif token in PUNCTUATION_INDEX:
            bits.extend((1, 1, 1, 0))
            _append_int(bits, PUNCTUATION_INDEX[token], 4)
        else:
            for character in token:
                raw = character.encode("utf-8")
                bits.extend((1, 1, 1, 1))
                _append_int(bits, len(raw), 5)
                for byte in raw:
                    _append_int(bits, byte, 8)
                raw_tokens += 1
    return {
        "bytes": _bits_to_bytes(bits),
        "bit_length": len(bits),
        "token_count": token_count,
        "dictionary_tokens": dictionary_tokens,
        "raw_tokens": raw_tokens,
        "lexicon_fingerprint": fingerprint,
    }


def decode_lexical_bits(data: bytes, bit_length: int, database_path: Path) -> str:
    words, _ = lexical_codebook(str(database_path.resolve()))
    bits = _bytes_to_bits(data)[:bit_length]
    output: list[str] = []
    offset = 0
    while offset < len(bits):
        first, offset = _read_int(bits, offset, 1)
        if first == 0:
            second, offset = _read_int(bits, offset, 1)
            width = 8 if second == 0 else 12
            word_id, offset = _read_int(bits, offset, width)
            output.append(words[word_id])
            continue
        second, offset = _read_int(bits, offset, 1)
        if second == 0:
            third, offset = _read_int(bits, offset, 1)
            if third == 0:
                word_id, offset = _read_int(bits, offset, 14)
                output.append(words[word_id])
                continue
            length, offset = _read_int(bits, offset, 6)
            characters = []
            for _ in range(length):
                value, offset = _read_int(bits, offset, 7)
                characters.append(chr(value))
            output.append("".join(characters))
            continue
        third, offset = _read_int(bits, offset, 1)
        fourth, offset = _read_int(bits, offset, 1)
        suffix = (third << 1) | fourth
        if suffix == 0:
            output.append(" ")
        elif suffix == 1:
            output.append("\n\n")
        elif suffix == 2:
            punctuation_id, offset = _read_int(bits, offset, 4)
            output.append(PUNCTUATION[punctuation_id])
        else:
            length, offset = _read_int(bits, offset, 5)
            raw = bytearray()
            for _ in range(length):
                value, offset = _read_int(bits, offset, 8)
                raw.append(value)
            output.append(bytes(raw).decode("utf-8"))
    return "".join(output)


def encode_lexical_payload(
    raw_text: str,
    database_path: Path,
    block_size: int = 32,
) -> dict[str, Any]:
    lexical = encode_lexical_bits(raw_text, database_path)
    encoded_bytes = lexical["bytes"]
    lexical_bits = _bytes_to_bits(encoded_bytes)[: int(lexical["bit_length"])]
    stream_crc = _crc16(raw_text.encode("utf-8"))
    protected_bits = lexical_bits + [
        (stream_crc >> shift) & 1 for shift in range(15, -1, -1)
    ]
    symbols = (
        MACRO_GUARD_START
        + _bits_to_macro_symbols(protected_bits)
        + MACRO_GUARD_END
    )
    blocks = [
        {
            "index": 0,
            "total": 1,
            "chunk_bytes": len(encoded_bytes),
            "crc16": stream_crc,
            "raw_bit_count": len(protected_bits),
            "encoded_bit_count": len(protected_bits),
            "symbols": symbols,
            "symbol_count": len(symbols),
        }
    ]
    return {
        "codebook_version": LEXICAL_PAYLOAD_VERSION,
        "lexicon_fingerprint": lexical["lexicon_fingerprint"],
        "codec": "tiered lexical IDs with UTF-8 escape",
        "ecc": "branch CRC-16 detection; optional parity branches reserved",
        "original_text": raw_text,
        "original_bytes": len(raw_text.encode("utf-8")),
        "encoded_bytes": len(encoded_bytes),
        "compressed_bytes": len(encoded_bytes),
        "lexical_bit_length": lexical["bit_length"],
        "token_count": lexical["token_count"],
        "dictionary_tokens": lexical["dictionary_tokens"],
        "raw_tokens": lexical["raw_tokens"],
        "payload_crc16": stream_crc,
        "block_size": len(encoded_bytes),
        "block_count": 1,
        "symbol_count": len(symbols),
        "blocks": blocks,
    }


def decode_lexical_payload(packet: dict[str, Any]) -> dict[str, Any]:
    database_path = Path(__file__).resolve().parent / "language_data" / "language.db"
    _, fingerprint = lexical_codebook(str(database_path.resolve()))
    guard_length = len(MACRO_GUARD_START)
    block = packet["blocks"][0]
    symbols = [int(item) for item in block["symbols"]]
    guards_valid = (
        symbols[:guard_length] == MACRO_GUARD_START
        and symbols[-guard_length:] == MACRO_GUARD_END
    )
    protected_length = int(packet["lexical_bit_length"]) + 16
    protected_bits = _macro_symbols_to_bits(
        symbols[guard_length:-guard_length],
        protected_length,
    )
    lexical_bits = protected_bits[: int(packet["lexical_bit_length"])]
    recorded_crc = 0
    for bit in protected_bits[int(packet["lexical_bit_length"]) :]:
        recorded_crc = (recorded_crc << 1) | bit
    encoded = _bits_to_bytes(lexical_bits)
    decode_error = None
    try:
        text = decode_lexical_bits(
            encoded,
            int(packet["lexical_bit_length"]),
            database_path,
        )
    except (IndexError, UnicodeDecodeError, ValueError) as exc:
        text = ""
        decode_error = f"{type(exc).__name__}: {exc}"
    crc_valid = (
        recorded_crc == int(packet["payload_crc16"])
        and _crc16(text.encode("utf-8")) == recorded_crc
    )
    lexicon_valid = fingerprint == packet.get("lexicon_fingerprint")
    return {
        "codebook_version": LEXICAL_PAYLOAD_VERSION,
        "text": text,
        "valid": guards_valid and crc_valid and lexicon_valid and decode_error is None,
        "blocks_valid": guards_valid and crc_valid,
        "payload_crc_valid": crc_valid,
        "lexicon_valid": lexicon_valid,
        "corrected_codewords": 0,
        "block_count": 1,
        "decode_error": decode_error,
    }


def encode_branch_protocol(header: dict[str, Any]) -> dict[str, Any]:
    fixed = struct.pack(
        BRANCH_FIXED_FORMAT,
        b"LB",
        10,
        bytes.fromhex(str(header["graph_fingerprint"])),
        bytes.fromhex(str(header["lexicon_fingerprint"])),
        int(header["lexical_bit_length"]),
        int(header["payload_crc16"]),
        int(header.get("flags", 1)),
    )
    raw = fixed + struct.pack(">H", _crc16(fixed))
    raw_bits = _bytes_to_bits(raw)
    symbols = COMPACT_GUARD_START + _bits_to_macro_symbols(raw_bits) + COMPACT_GUARD_END
    return {
        "codebook_version": BRANCH_PROTOCOL_VERSION,
        "header": header,
        "raw_bit_count": len(raw_bits),
        "encoded_bit_count": len(raw_bits),
        "symbols": symbols,
        "symbol_count": len(symbols),
        "crc16": _crc16(fixed),
    }


def decode_branch_protocol(packet: dict[str, Any]) -> dict[str, Any]:
    symbols = [int(item) for item in packet["symbols"]]
    guard_length = len(COMPACT_GUARD_START)
    guards_valid = (
        symbols[:guard_length] == COMPACT_GUARD_START
        and symbols[-guard_length:] == COMPACT_GUARD_END
    )
    raw = _bits_to_bytes(
        _macro_symbols_to_bits(
            symbols[guard_length:-guard_length],
            int(packet["raw_bit_count"]),
        )
    )
    fixed, recorded_crc = raw[:-2], struct.unpack(">H", raw[-2:])[0]
    values = struct.unpack(BRANCH_FIXED_FORMAT, fixed)
    (
        magic,
        version,
        graph_fingerprint,
        lexicon_fingerprint,
        lexical_bit_length,
        payload_crc16,
        flags,
    ) = values
    header = {
        "version": "0.9.1",
        "graph_fingerprint": graph_fingerprint.hex(),
        "lexicon_fingerprint": lexicon_fingerprint.hex(),
        "lexical_bit_length": lexical_bit_length,
        "payload_crc16": payload_crc16,
        "flags": flags,
    }
    return {
        "codebook_version": BRANCH_PROTOCOL_VERSION,
        "header": header,
        "valid": guards_valid and magic == b"LB" and version == 10 and recorded_crc == _crc16(fixed),
        "guards_valid": guards_valid,
        "crc_valid": recorded_crc == _crc16(fixed),
        "corrected_codewords": 0,
    }
