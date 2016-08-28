function [lows,highs]=bounding_box(vox)
    lows=min(vox);
    highs=max(vox);
