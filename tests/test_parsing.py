import re

from loglens.masking import mask_text
from loglens.parse import template_regex
from loglens.parse_eval import grouping_accuracy, gt_regex


def java_to_py(rx: str) -> str:
    # \Q...\E quoting is Java-only; translate it so Python can check the same regex.
    return re.sub(r"\\Q(.*?)\\E", lambda m: re.escape(m.group(1)), rx)


def test_mask_ip_with_port_and_leading_slash():
    out = mask_text("Receiving block blk_-160 src: /10.250.19.102:54106 dest: /10.250.19.102:50010")
    assert out == "Receiving block <BLK> src: <IP> dest: <IP>"


def test_mask_path_swallows_block_id_inside_it():
    out = mask_text("Deleting block blk_123 file /mnt/hadoop/dfs/data/current/subdir54/blk_123")
    assert out == "Deleting block <BLK> file <PATH>"


def test_mask_numbers_but_not_inside_words():
    out = mask_text("PacketResponder 1 for block blk_9 terminating size 67108864 dfs.DataNode2")
    assert out == "PacketResponder <NUM> for block <BLK> terminating size <NUM> dfs.DataNode2"


def test_masking_is_idempotent():
    once = mask_text("BLOCK* ask 10.250.14.224:50010 to replicate blk_-1 to datanode(s) 10.251.1.2:50010")
    assert mask_text(once) == once


def test_template_regex_wildcard_is_one_token():
    rx = re.compile(java_to_py(template_regex("PacketResponder <*> for block <BLK> terminating")))
    assert rx.match("PacketResponder <NUM> for block <BLK> terminating")
    assert not rx.match("PacketResponder <NUM> <NUM> for block <BLK> terminating")


def test_gt_regex_allows_anything_at_wildcards():
    rx = re.compile(java_to_py(gt_regex("[*]Receiving block[*]src:[*]dest:[*]")))
    assert rx.match("Receiving block blk_1 src: /1.2.3.4:5 dest: /1.2.3.4:6")
    assert not rx.match("Received block blk_1 src: a dest: b")


def test_grouping_accuracy_perfect_one_to_one():
    cells = [("T01", "E5", 10), ("T02", "E11", 5)]
    r = grouping_accuracy(cells)
    assert r["grouping_accuracy"] == 1.0


def test_grouping_accuracy_merge_split_and_unk_are_wrong():
    cells = [
        ("T01", "E5", 10),   # T01 also holds E6 lines: a merge, both wrong
        ("T01", "E6", 4),
        ("T02", "E11", 3),   # E11 is split across T02 and T03: both wrong
        ("T03", "E11", 2),
        ("T04", "E9", 6),    # clean one-to-one: correct
        ("T05", "E2", 7),    # E2 also has one UNK line, so T05's group is incomplete: wrong
        ("UNK", "E2", 1),
    ]
    r = grouping_accuracy(cells)
    assert r["correct"] == 6
    assert r["lines"] == 33
    assert r["templates_mixing_gt"] == 1
    assert r["gt_split_across_templates"] == 1


def test_bgl_masks_node_ids_and_both_hex_styles():
    out = mask_text("iar 00106210 dear 0244c20c at R02-M1-N0-C:J12-U11 addr 0x00004ed8 count 7", "bgl")
    assert out == "iar <HEX> dear <HEX> at <NODE> addr <HEX> count <NUM>"


def test_bgl_masks_keep_words_with_digits():
    out = mask_text("ciod: Received signal 15, code=0, errno=0", "bgl")
    assert out == "ciod: Received signal <NUM>, code=<NUM>, errno=<NUM>"


def test_scheme_settings_inherit_from_their_log():
    from loglens.config import parent, session_col, settings

    s = settings("bgl_20ev")
    assert parent("bgl_20ev") == "bgl" and parent("hdfs") == "hdfs"
    assert s["session"] == {"kind": "count", "size": 20}
    assert s["masks"] == "bgl" and s["expected_lines"] == 4_747_963
    assert session_col("bgl_1h") == "session_id" and session_col("hdfs") == "block_id"


def test_thunderbird_masks_hosts_but_not_ordinary_words():
    out = mask_text("xinetd[4019]: START: rsync pid=5681 from=10.100.28.42 on an214 and tbird-sm1, eth0 up", "thunderbird")
    assert out == "xinetd[<NUM>]: START: rsync pid=<NUM> from=<IP> on <NODE> and <NODE>, eth0 up"
