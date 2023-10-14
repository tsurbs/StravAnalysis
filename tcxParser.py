import xml.etree.ElementTree as ET
tree = ET.parse('test_run.tcx')
root = tree.getroot()

print(root)