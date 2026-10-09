"""
Dhara - Priority Ranking Engine
Calculates actionable priority: Rank = Savings × Urgency × Ease of Fix
Selects the Top 3 High-Impact Executive Actions for Dr. Sharma.
"""

class PriorityEngine:
    def __init__(self, config=None):
        self.config = config or {}
        weights = self.config.get("weights", {}).get("priority", {})
        self.w_savings = weights.get("savings", 0.40)
        self.w_urgency = weights.get("urgency", 0.35)
        self.w_ease = weights.get("ease_of_fix", 0.25)

    def rank_recommendations(self, recommendations):
        """
        Ranks recommendations using:
        Rank Metric = Savings Factor × Urgency Score × Ease-of-Fix Score
        
        Also generates executive summary:
        "Fix 3 issues to save ₹X this month"
        """
        if not recommendations:
            return [], {"total_top3_savings_inr": 0, "total_top3_co2_kg": 0, "executive_headline": "No active issues detected."}

        scored_recs = []
        max_savings = max((r.get("estimated_monthly_savings_inr", 1.0) for r in recommendations), default=1.0)
        max_savings = max(1.0, max_savings)

        for r in recommendations:
            r_copy = dict(r)
            savings_val = r_copy.get("estimated_monthly_savings_inr", 0.0)
            urgency = r_copy.get("urgency_score", 5)       # 1 - 10
            ease = r_copy.get("ease_of_fix_score", 5)        # 1 - 10
            
            # Normalized savings factor (scale 1 to 10)
            savings_factor = 1.0 + 9.0 * (savings_val / max_savings)
            
            # Exact formula from specification: Rank = savings x urgency x ease of fix
            rank_score = round(savings_factor * urgency * ease, 1)
            
            # Weighted multi-criteria score for secondary sorting
            weighted_score = round(
                (savings_factor * 10 * self.w_savings) +
                (urgency * 10 * self.w_urgency) +
                (ease * 10 * self.w_ease),
                1
            )
            
            r_copy["savings_factor"] = round(savings_factor, 1)
            r_copy["rank_score"] = rank_score
            r_copy["weighted_score"] = weighted_score
            scored_recs.append(r_copy)

        # Sort descending by rank score
        ranked = sorted(scored_recs, key=lambda x: (x["rank_score"], x["weighted_score"]), reverse=True)
        
        # Assign ranking index (1, 2, 3...)
        for i, r in enumerate(ranked):
            r["rank_position"] = i + 1

        top_3 = ranked[:3]
        total_top3_savings = sum(r.get("estimated_monthly_savings_inr", 0.0) for r in top_3)
        total_top3_co2 = sum(r.get("estimated_monthly_co2_kg", 0.0) for r in top_3)
        
        executive_headline = f"Fix top 3 priority issues to save ₹{int(total_top3_savings):,} this month and avoid {int(total_top3_co2):,} kg CO₂"

        summary = {
            "top_3": top_3,
            "total_top3_savings_inr": round(total_top3_savings, 0),
            "total_top3_co2_kg": round(total_top3_co2, 1),
            "executive_headline": executive_headline,
            "total_action_count": len(ranked)
        }

        return ranked, summary
