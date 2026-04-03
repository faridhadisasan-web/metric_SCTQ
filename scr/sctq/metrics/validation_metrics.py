import math
import numpy as np
from typing import Dict, Any, List, Tuple
from sctq.types import TrackSet
from sctq.synthetic.object_models import SyntheticObject
from scipy.optimize import linear_sum_assignment

def compute_synthetic_validation(trackset: TrackSet, gt_objects: List[SyntheticObject]) -> Dict[str, Any]:
    """Compute strong GT validation metrics (ID Precision, ID Recall, ID F1) for synthetic scenes."""
    metrics = {}

    gt_tracks_count = len(gt_objects)
    metrics['gt_track_count'] = gt_tracks_count

    pred_tracks_count = len(trackset.tracks)
    metrics['pred_track_count'] = pred_tracks_count

    if pred_tracks_count == 0 or gt_tracks_count == 0:
        return {
            'gt_track_count': gt_tracks_count,
            'pred_track_count': pred_tracks_count,
            'idp': 0.0,
            'idr': 0.0,
            'idf1': 0.0,
            'gt_fragmentation': 0,
            'actual_assignment_purity': 0.0
        }

    # We will compute ID matches frame by frame to find total true positives (IDTP)
    # Then IDP = IDTP / IDTP + IDFP (all pred pts)
    # IDR = IDTP / IDTP + IDFN (all gt pts)

    total_gt_pts = 0
    total_pred_pts = 0

    # Pre-organize GT and Pred points by frame
    gt_by_frame = {}
    for gt_obj in gt_objects:
        for pt in gt_obj.trajectory:
            total_gt_pts += 1
            f = pt.frame_idx
            if f not in gt_by_frame:
                gt_by_frame[f] = []
            gt_by_frame[f].append((gt_obj.object_id, pt))

    pred_by_frame = {}
    for tid, t in trackset.tracks.items():
        for pt in t.points:
            total_pred_pts += 1
            f = pt.frame_idx
            if f not in pred_by_frame:
                pred_by_frame[f] = []
            pred_by_frame[f].append((tid, pt))

    # Frame-wise assignment
    # Track which pred_id is mostly assigned to which gt_id
    assignment_votes = {} # (gt_id, pred_id) -> count

    for f in gt_by_frame.keys():
        if f not in pred_by_frame:
            continue

        gts = gt_by_frame[f]
        preds = pred_by_frame[f]

        # Distance matrix
        D = np.zeros((len(gts), len(preds)), dtype=np.float32)
        for i, (g_id, g_pt) in enumerate(gts):
            for j, (p_id, p_pt) in enumerate(preds):
                D[i, j] = math.hypot(g_pt.cx - p_pt.cx, g_pt.cy - p_pt.cy)

        # Assignment
        row_ind, col_ind = linear_sum_assignment(D)

        for r, c in zip(row_ind, col_ind):
            if D[r, c] < 50.0: # Valid match
                g_id = gts[r][0]
                p_id = preds[c][0]
                pair = (g_id, p_id)
                assignment_votes[pair] = assignment_votes.get(pair, 0) + 1

    # Now assign each GT trajectory to EXACTLY ONE Pred trajectory (the one with most votes)
    # This is a simplified ID assignment strategy, giving a solid approximation of IDF1

    # We must bipartite match GT tracks to Pred tracks globally based on vote counts
    gt_ids = [g.object_id for g in gt_objects]
    pred_ids = list(trackset.tracks.keys())

    Global_D = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
    for i, g_id in enumerate(gt_ids):
        for j, p_id in enumerate(pred_ids):
            # Cost is negative votes
            Global_D[i, j] = -assignment_votes.get((g_id, p_id), 0)

    row_ind, col_ind = linear_sum_assignment(Global_D)

    idtp = 0
    assigned_gt_to_pred = {}
    for r, c in zip(row_ind, col_ind):
        votes = -Global_D[r, c]
        if votes > 0:
            idtp += votes
            assigned_gt_to_pred[gt_ids[r]] = pred_ids[c]

    idp = idtp / total_pred_pts if total_pred_pts > 0 else 0.0
    idr = idtp / total_gt_pts if total_gt_pts > 0 else 0.0
    idf1 = 2 * idp * idr / (idp + idr) if (idp + idr) > 0 else 0.0

    metrics['idp'] = float(idp)
    metrics['idr'] = float(idr)
    metrics['idf1'] = float(idf1)

    # Fragmentation truth: how many DIFFERENT pred tracks were strongly assigned to a single GT track?
    # Count any pred track that got at least 5 frames of overlap with the GT
    gt_fragmentation = 0
    for g_id in gt_ids:
        assigned_preds = [p_id for (g, p_id), v in assignment_votes.items() if g == g_id and v >= 5]
        if len(assigned_preds) > 1:
            gt_fragmentation += (len(assigned_preds) - 1)

    metrics['gt_fragmentation'] = gt_fragmentation

    # Purity: how many pred tracks tracked ONLY ONE GT?
    pure_preds = 0
    for p_id in pred_ids:
        assigned_gts = [g for (g, p), v in assignment_votes.items() if p == p_id and v >= 5]
        if len(assigned_gts) == 1:
            pure_preds += 1

    metrics['actual_assignment_purity'] = pure_preds / pred_tracks_count if pred_tracks_count > 0 else 0.0

    return metrics
