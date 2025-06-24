import cadquery as cq
import trimesh
import numpy as np
from typing import Tuple, Optional
from loguru import logger
import tempfile
import multiprocessing
from functools import partial


class CADMetrics:
    """Advanced CAD-specific evaluation metrics"""
    
    @staticmethod
    def validate_syntax(code: str) -> bool:
        """Check if code compiles and produces valid CAD object"""
        try:
            loc = {}
            exec(code, {"cq": cq}, loc)
            return isinstance(loc.get("result", None), cq.Workplane)
        except Exception as e:
            logger.debug(f"Syntax error: {e}")
            return False
    
    @staticmethod
    def calculate_mesh_iou(code1: str, code2: str, samples: int = 10000) -> Optional[float]:
        """Calculate mesh IoU using point sampling"""
        try:
            # Execute both codes
            loc1, loc2 = {}, {}
            exec(code1, {"cq": cq}, loc1)
            exec(code2, {"cq": cq}, loc2)
            
            # Export to temporary files
            with tempfile.NamedTemporaryFile(suffix='.stl') as f1, \
                 tempfile.NamedTemporaryFile(suffix='.stl') as f2:
                
                cq.exporters.export(loc1["result"], f1.name)
                cq.exporters.export(loc2["result"], f2.name)
                
                # Load meshes
                mesh1 = trimesh.load(f1.name)
                mesh2 = trimesh.load(f2.name)
                
                # Find common bounding box
                bounds = np.vstack([mesh1.bounds, mesh2.bounds])
                min_bound = bounds.min(axis=0)
                max_bound = bounds.max(axis=0)
                
                # Sample points
                points = np.random.uniform(
                    low=min_bound,
                    high=max_bound,
                    size=(samples, 3)
                
                # Calculate containment
                in1 = mesh1.contains(points)
                in2 = mesh2.contains(points)
                
                intersection = np.sum(in1 & in2)
                union = np.sum(in1 | in2)
                
                return intersection / union if union > 0 else 0.0
                
        except Exception as e:
            logger.warning(f"Mesh comparison failed: {e}")
            return None
    
    @staticmethod
    def batch_evaluate(generated: List[str], references: List[str]) -> Dict[str, float]:
        """Parallel evaluation of multiple samples"""
        with multiprocessing.Pool() as pool:
            syntax_results = pool.map(CADMetrics.validate_syntax, generated)
            iou_results = pool.starmap(
                CADMetrics.calculate_mesh_iou,
                zip(generated, references)
            )
        
        return {
            "syntax_accuracy": np.mean(syntax_results),
            "mean_iou": np.nanmean([x for x in iou_results if x is not None])
        }