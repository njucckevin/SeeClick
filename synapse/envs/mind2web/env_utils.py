from lxml import etree
import copy
import re
import json
import os
import string


def load_json(data_dir, folder_name):
    folder_path = os.path.join(data_dir, folder_name)
    print(f"Data path: {folder_path}")
    data_paths = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.endswith(".json")
    ]
    data_paths = sorted(data_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # Construct trajectory dataset
    samples = []
    for data_path in data_paths:
        with open(data_path, "r") as f:
            samples.extend(json.load(f))
    print("# of samples:", len(samples))

    return samples


def get_target_obs(dom_tree, target_element_ids):
    pruned_tree = prune_tree(dom_tree, target_element_ids)
    tree_repr, _ = get_tree_repr(pruned_tree, id_mapping={}, keep_html_brackets=True)

    return tree_repr


def get_target_act(example, target_element_id):
    action_op = example["operation"]["op"]
    action_value = example["operation"]["value"]
    target_action = f"{action_op} [{target_element_id}]"
    if action_op != "CLICK":
        target_action += f" [{action_value}]"

    return target_action


def parse_act_str(act_str):
    # Compile the regular expression pattern
    pattern = re.compile(r"(?:^|\s)(CLICK|SELECT|TYPE)?\s?\[(.+?)\](?:\s\[(.+?)\])?")
    # Search for the pattern in the string
    match = pattern.search(act_str)
    if match:
        # Extract the matching groups
        action_op = match.group(1)  # This will be None if not in the list
        target_element_id = match.group(2)
        action_value = match.group(3)  # This will be None if not present
        return action_op, target_element_id, action_value
    else:
        return None, None, None


def construct_act_str(op, val):
    if op is None:
        if val is None:
            return " "
        return " " + val
    if op == "CLICK" or val is None:
        return op + " "
    return f"{op} {val}"


def get_target_obs_and_act(example):
    if len(example["pos_candidates"]) == 0:
        # Simplify the raw_html if pos_candidates is empty (not in the cleaned html)
        dom_tree = etree.fromstring(example["raw_html"])
        gt_element = dom_tree.xpath(
            f"//*[@data_pw_testid_buckeye='{example['action_uid']}']"
        )
        element_id = gt_element[0].get("backend_node_id")
        raw_obs = get_target_obs(dom_tree, [element_id])
        # Find the start index of the target element using the element ID
        start_idx = raw_obs.find(f"id={element_id}")
        # Find the start tag for the target element
        start_tag_idx = raw_obs.rfind("<", 0, start_idx)
        end_tag_idx = raw_obs.find(">", start_idx)
        # Extract the tag name
        tag_name = raw_obs[start_tag_idx + 1 : end_tag_idx].split()[0]
        # Initialize count for open and close tags
        open_count = 0
        close_count = 0
        search_idx = start_tag_idx
        while True:
            # Find the next open or close tag of the same type
            next_open_tag = raw_obs.find(f"<{tag_name}", search_idx)
            next_close_tag = raw_obs.find(f"</{tag_name}>", search_idx)
            # No more tags found, break
            if next_open_tag == -1 and next_close_tag == -1:
                break
            # Decide whether the next tag is an open or close tag
            if next_open_tag != -1 and (
                next_open_tag < next_close_tag or next_close_tag == -1
            ):
                open_count += 1
                search_idx = raw_obs.find(">", next_open_tag) + 1
            else:
                close_count += 1
                search_idx = next_close_tag + len(f"</{tag_name}>")
            # If we've closed all open tags, break
            if open_count == close_count:
                break
        # Extract the target element
        o = f"<html> {raw_obs[start_tag_idx:search_idx]} </html>"
        a = get_target_act(example, element_id)
    else:
        dom_tree = etree.fromstring(example["cleaned_html"])
        element_id = example["pos_candidates"][0]["backend_node_id"]
        o = get_target_obs(dom_tree, [element_id])
        a = get_target_act(example, element_id)

    return o, a


def get_top_k_obs(s: dict, top_k: int, use_raw: bool = True) -> tuple[str, str]:
    # Find one positive candidate (it can be zero)
    pos_candidates = s["pos_candidates"]
    pos_ids = [c["backend_node_id"] for c in pos_candidates][:1]
    # Find top_k - 1 negative candidates
    neg_candidates = s["neg_candidates"]
    neg_candidates = sorted(neg_candidates, key=lambda c: c["rank"])[: top_k - 1]
    neg_ids = [c["backend_node_id"] for c in neg_candidates]
    # Prune html with all candidates
    all_candidates = pos_ids + neg_ids
    obs = get_target_obs(etree.fromstring(s["cleaned_html"]), all_candidates)
    # If there is no positive candidate in cleaned_html, get it from raw_html
    if len(s["pos_candidates"]) == 0:
        assert use_raw
        # Simplify the raw_html if pos_candidates is empty (not in the cleaned html)
        dom_tree = etree.fromstring(s["raw_html"])
        gt_element = dom_tree.xpath(f"//*[@data_pw_testid_buckeye='{s['action_uid']}']")
        element_id = gt_element[0].get("backend_node_id")
        raw_obs = get_target_obs(dom_tree, [element_id])
        # Find the start index of the target element using the element ID
        start_idx = raw_obs.find(f"id={element_id}")
        # Find the start tag for the target element
        start_tag_idx = raw_obs.rfind("<", 0, start_idx)
        end_tag_idx = raw_obs.find(">", start_idx)
        # Extract the tag name
        tag_name = raw_obs[start_tag_idx + 1 : end_tag_idx].split()[0]
        # Initialize count for open and close tags
        open_count = 0
        close_count = 0
        search_idx = start_tag_idx
        while True:
            # Find the next open or close tag of the same type
            next_open_tag = raw_obs.find(f"<{tag_name}", search_idx)
            next_close_tag = raw_obs.find(f"</{tag_name}>", search_idx)
            # No more tags found, break
            if next_open_tag == -1 and next_close_tag == -1:
                break
            # Decide whether the next tag is an open or close tag
            if next_open_tag != -1 and (
                next_open_tag < next_close_tag or next_close_tag == -1
            ):
                open_count += 1
                search_idx = raw_obs.find(">", next_open_tag) + 1
            else:
                close_count += 1
                search_idx = next_close_tag + len(f"</{tag_name}>")
            # If we've closed all open tags, break
            if open_count == close_count:
                break
        # Extract the target element
        target_element = raw_obs[start_tag_idx:search_idx]
        obs = obs.replace("</html>", f"{target_element} </html>")

    return obs, all_candidates


def calculate_f1(pred, label):
    pred = set(pred.strip().split())
    label = set(label.strip().split())
    # remove punctuation
    pred = set([x for x in pred if x not in string.punctuation])
    label = set([x for x in label if x not in string.punctuation])
    if len(pred) == 0 and len(label) == 0:
        return 1
    if len(pred) == 0 or len(label) == 0:
        return 0

    tp = len(pred & label)
    fp = len(pred - label)
    fn = len(label - pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def get_descendants(node, max_depth, current_depth=0):
    if current_depth > max_depth:
        return []

    descendants = []
    for child in node:
        descendants.append(child)
        descendants.extend(get_descendants(child, max_depth, current_depth + 1))

    return descendants


def get_attribute_repr(node, max_value_length=5, max_length=20):
    # get attribute values in order
    attr_values_set = set()
    attr_values = ""
    for attr in [
        "role",
        "aria_role",
        "type",
        "alt",
        "aria_description",
        "aria_label",
        "label",
        "title",
        "name",
        "text_value",
        "value",
        "placeholder",
        "input_checked",
        "input_value",
        "option_selected",
        "class",
    ]:
        if attr in node.attrib and node.attrib[attr] is not None:
            value = node.attrib[attr].lower()
            # less menaingful values
            if value in [
                "hidden",
                "none",
                "presentation",
                "null",
                "undefined",
            ] or value.startswith("http"):
                continue
            value = value.split()
            value = " ".join([v for v in value if len(v) < 15][:max_value_length])
            if value and value not in attr_values_set:
                attr_values_set.add(value)
                attr_values += value + " "
    uid = node.attrib.get("backend_node_id", "")
    # clear all attributes
    node.attrib.clear()
    if uid:
        node.attrib["id"] = uid
    # add meta attribute
    if attr_values:
        node.attrib["meta"] = " ".join(attr_values.split()[:max_length])


def prune_tree(
    dom_tree,
    candidate_set,
    max_depth=5,
    max_children=50,
    max_sibling=3,
):
    nodes_to_keep = set()
    for candidate_id in candidate_set:
        candidate_node = dom_tree.xpath(f'//*[@backend_node_id="{candidate_id}"]')[0]
        nodes_to_keep.add(candidate_node.attrib["backend_node_id"])
        # get all ancestors
        nodes_to_keep.update(
            [
                x.attrib.get("backend_node_id", "")
                for x in candidate_node.xpath("ancestor::*")
            ]
        )
        # get descendants with max depth
        nodes_to_keep.update(
            [
                x.attrib.get("backend_node_id", "")
                for x in get_descendants(candidate_node, max_depth)
            ][:max_children]
        )
        # get siblings within range
        parent = candidate_node.getparent()
        if parent is not None:
            siblings = [x for x in parent.getchildren() if x.tag != "text"]
            idx_in_sibling = siblings.index(candidate_node)
            nodes_to_keep.update(
                [
                    x.attrib.get("backend_node_id", "")
                    for x in siblings[
                        max(0, idx_in_sibling - max_sibling) : idx_in_sibling
                        + max_sibling
                        + 1
                    ]
                ]
            )
    # clone the tree
    new_tree = copy.deepcopy(dom_tree)
    # remove nodes not in nodes_to_keep
    for node in new_tree.xpath("//*")[::-1]:
        if node.tag != "text":
            is_keep = node.attrib.get("backend_node_id", "") in nodes_to_keep
            is_candidate = node.attrib.get("backend_node_id", "") in candidate_set
        else:
            is_keep = (
                node.getparent().attrib.get("backend_node_id", "") in nodes_to_keep
            )
            is_candidate = (
                node.getparent().attrib.get("backend_node_id", "") in candidate_set
            )
        if not is_keep and node.getparent() is not None:
            node.getparent().remove(node)
        else:
            if not is_candidate or node.tag == "text":
                node.attrib.pop("backend_node_id", None)
            if (
                len(node.attrib) == 0
                and not any([x.tag == "text" for x in node.getchildren()])
                and node.getparent() is not None
                and node.tag != "text"
                and len(node.getchildren()) <= 1
            ):
                # insert all children into parent
                for child in node.getchildren():
                    node.addprevious(child)
                node.getparent().remove(node)
    return new_tree


def get_tree_repr(
    tree, max_value_length=5, max_length=20, id_mapping={}, keep_html_brackets=False
):
    if isinstance(tree, str):
        tree = etree.fromstring(tree)
    else:
        tree = copy.deepcopy(tree)
    for node in tree.xpath("//*"):
        if node.tag != "text":
            if "backend_node_id" in node.attrib:
                if node.attrib["backend_node_id"] not in id_mapping:
                    id_mapping[node.attrib["backend_node_id"]] = len(id_mapping)
                # node.attrib["backend_node_id"] = str(
                #     id_mapping[node.attrib["backend_node_id"]]
                # )
            get_attribute_repr(node, max_value_length, max_length)
        else:
            node.text = " ".join(node.text.split()[:max_length])
    tree_repr = etree.tostring(tree, encoding="unicode")

    tree_repr = tree_repr.replace('"', " ")
    tree_repr = (
        tree_repr.replace("meta= ", "").replace("id= ", "id=").replace(" >", ">")
    )
    tree_repr = re.sub(r"<text>(.*?)</text>", r"\1", tree_repr)
    if not keep_html_brackets:
        tree_repr = tree_repr.replace("/>", "$/$>")
        tree_repr = re.sub(r"</(.+?)>", r")", tree_repr)
        tree_repr = re.sub(r"<(.+?)>", r"(\1", tree_repr)
        tree_repr = tree_repr.replace("$/$", ")")

    html_escape_table = [
        ("&quot;", '"'),
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&nbsp;", " "),
        ("&ndash;", "-"),
        ("&rsquo;", "'"),
        ("&lsquo;", "'"),
        ("&ldquo;", '"'),
        ("&rdquo;", '"'),
        ("&#39;", "'"),
        ("&#40;", "("),
        ("&#41;", ")"),
    ]
    for k, v in html_escape_table:
        tree_repr = tree_repr.replace(k, v)
    tree_repr = re.sub(r"\s+", " ", tree_repr).strip()

    return tree_repr, id_mapping
