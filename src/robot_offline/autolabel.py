"""Classical body segmentation for draft labels and cut-outs.

Subtracts the session's empty-table frame (Lab distance), opens with a kernel of half the smallest
body width to remove leads, and traces external contours. Drafts go into the manifest with
label_source "auto". Reliable on spread-out scenes; merged blobs need fixing by hand.
"""
