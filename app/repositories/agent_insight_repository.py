from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any

from app.extensions import db
from app.models.agent_insight import AgentInsight


class AgentInsightRepository:
    """Repository for AgentInsight model operations."""

    def save(self, insight: AgentInsight) -> AgentInsight:
        """Save insight to database."""
        db.session.add(insight)
        db.session.commit()
        return insight

    def get_by_id(self, insight_id: int) -> Optional[AgentInsight]:
        """Get insight by ID."""
        return db.session.get(AgentInsight, insight_id)

    def find_by_agent(self, agent_name: str, limit: int = 100) -> List[AgentInsight]:
        """Find insights for a specific agent."""
        return (
            AgentInsight.query.filter_by(agent_name=agent_name)
            .order_by(AgentInsight.created_at.desc())
            .limit(limit)
            .all()
        )

    def find_by_type(self, insight_type: str, limit: int = 100) -> List[AgentInsight]:
        """Find insights by type."""
        return (
            AgentInsight.query.filter_by(insight_type=insight_type)
            .order_by(AgentInsight.created_at.desc())
            .limit(limit)
            .all()
        )

    def find_unapplied(self, agent_name: Optional[str] = None) -> List[AgentInsight]:
        """Find insights that haven't been applied yet."""
        query = AgentInsight.query.filter_by(applied_at=None)
        if agent_name:
            query = query.filter_by(agent_name=agent_name)
        return query.order_by(AgentInsight.created_at.desc()).all()

    def find_recent(self, hours: int = 24, limit: int = 100) -> List[AgentInsight]:
        """Find recent insights."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return (
            AgentInsight.query.filter(AgentInsight.created_at >= cutoff)
            .order_by(AgentInsight.created_at.desc())
            .limit(limit)
            .all()
        )

    def get_statistics(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get insight statistics."""
        query = AgentInsight.query
        if agent_name:
            query = query.filter_by(agent_name=agent_name)
        
        insights = query.all()
        
        if not insights:
            return {"total": 0, "by_type": {}, "applied_rate": 0}
        
        by_type = {}
        applied_count = 0
        
        for insight in insights:
            by_type[insight.insight_type] = by_type.get(insight.insight_type, 0) + 1
            if insight.applied_at:
                applied_count += 1
        
        applied_rate = applied_count / len(insights) if insights else 0
        
        return {
            "total": len(insights),
            "by_type": by_type,
            "applied_count": applied_count,
            "applied_rate": applied_rate,
            "unapplied_count": len(insights) - applied_count,
        }
    
    def cleanup_old_insights(self, days: int = 90) -> int:
        """Clean up insights older than specified days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        deleted_count = AgentInsight.query.filter(
            AgentInsight.created_at < cutoff
        ).delete()
        db.session.commit()
        return deleted_count
