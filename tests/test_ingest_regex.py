import re

from loglens.config import DATASETS

HDFS = re.compile(DATASETS["hdfs"]["regex"])


def test_hdfs_line_parses_into_six_fields():
    line = (
        "081109 203615 148 INFO dfs.DataNode$PacketResponder: "
        "PacketResponder 1 for block blk_38865049064139660 terminating"
    )
    m = HDFS.match(line)
    assert m is not None
    date, time, pid, level, component, message = m.groups()
    assert (date, time, pid, level) == ("081109", "203615", "148", "INFO")
    assert component == "dfs.DataNode$PacketResponder"
    assert message.endswith("terminating")


def test_hdfs_message_with_colons_keeps_full_message():
    line = (
        "081109 204005 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: "
        "blockMap updated: 10.251.73.220:50010 is added to blk_7128370237687728475 size 67108864"
    )
    m = HDFS.match(line)
    assert m is not None
    assert m.group(5) == "dfs.FSNamesystem"
    assert m.group(6).startswith("BLOCK* NameSystem.addStoredBlock:")


def test_garbage_line_does_not_parse():
    assert HDFS.match("not a log line") is None
