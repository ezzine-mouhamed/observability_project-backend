"""create initial tables

Revision ID: initial_create_tables
Revises: 
Create Date: 2024-02-07 16:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'initial_create_tables'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create tasks table
    op.create_table('tasks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('task_type', sa.String(length=100), nullable=False),
        sa.Column('input_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('parameters', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('output_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='pending'),
        sa.Column('execution_time_ms', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('complexity_level', sa.String(length=50), nullable=True),
        sa.Column('agent_involved', sa.String(length=100), nullable=True),
        sa.Column('validation_results', postgresql.JSONB(astext_type=sa.Text()), nullable=True, server_default='{}'),
        sa.Column('performance_metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=True, server_default='{}'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create execution_traces table
    op.create_table('execution_traces',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('trace_id', sa.String(length=36), nullable=False),
        sa.Column('parent_trace_id', sa.String(length=36), nullable=True),
        sa.Column('task_id', sa.Integer(), nullable=True),
        sa.Column('operation', sa.String(length=200), nullable=False),
        sa.Column('context', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('decisions', postgresql.JSONB(astext_type=sa.Text()), nullable=True, server_default='[]'),
        sa.Column('events', postgresql.JSONB(astext_type=sa.Text()), nullable=True, server_default='[]'),
        sa.Column('agent_observations', postgresql.JSONB(astext_type=sa.Text()), nullable=True, server_default='[]'),
        sa.Column('quality_metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=True, server_default='{}'),
        sa.Column('agent_context', postgresql.JSONB(astext_type=sa.Text()), nullable=True, server_default='{}'),
        sa.Column('error', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('start_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('end_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_ms', sa.Integer(), nullable=True),
        sa.Column('success', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('performance_score', sa.Float(), nullable=True),
        sa.Column('complexity_score', sa.Float(), nullable=True),
        sa.Column('efficiency_score', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('trace_id'),
        sa.ForeignKeyConstraint(['task_id'], ['tasks.id'], ),
        sa.Index('ix_execution_traces_task_id', 'task_id'),
        sa.Index('ix_execution_traces_operation', 'operation'),
        sa.Index('ix_execution_traces_start_time', 'start_time')
    )


def downgrade():
    op.drop_table('execution_traces')
    op.drop_table('tasks')