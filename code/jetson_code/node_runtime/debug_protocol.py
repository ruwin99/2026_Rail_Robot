"""모터·센서 디버깅이 공유하는 READY 메타데이터와 명시적 보정 단위."""


def parse_ready(fields, capability):
    """Return seven numeric fields and physical-right forward trim in percent.

    방향별 보정 펌웨어도 기존 READY 길이를 유지한다. 후진 trim은 이 프레임에 없다.
    """
    explicit_bp = len(fields) == 10 and fields[-1] == 'BP'
    if fields[0] != 'DBG1_READY' or (len(fields) != 9 and not explicit_bp):
        raise ValueError('Unexpected DBG1_READY format/unit')
    if not all(v.isascii() and v.isdecimal() for v in fields[2:9]):
        raise ValueError('Malformed READY metadata')
    values = tuple(map(int, fields[2:9]))
    if values[0] != capability or any(not 0 < v <= 0xffffffff for v in values[1:]):
        raise ValueError('Firmware capability or PWM/counter metadata mismatch')
    trim = values[2]
    if explicit_bp:
        if trim > 20000:
            raise ValueError('RIGHT_MOTOR_TRIM_BP exceeds 20000')
        trim_percent = trim / 100.0
    else:
        # 구형만 기존 percent/permille 해석을 유지한다. 새 BP는 반드시 태그가 필요하다.
        if trim > 2000:
            raise ValueError('Missing BP unit or invalid legacy trim')
        trim_percent = trim / 10.0 if trim > 500 else float(trim)
    return values, trim_percent
