import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json


class CreditVisualizationDashboard:
    def __init__(self):
        self.color_scheme = {
            'excellent': '#10B981',
            'good': '#3B82F6',
            'fair': '#F59E0B',
            'poor': '#EF4444',
            'background': '#F9FAFB',
            'text': '#1F2937',
            'grid': '#E5E7EB'
        }
        
        self.layout_template = {
            'plot_bgcolor': self.color_scheme['background'],
            'paper_bgcolor': 'white',
            'font': {'family': 'system-ui', 'color': self.color_scheme['text']},
            'margin': {'l': 40, 'r': 40, 't': 60, 'b': 40}
        }
    
    def create_credit_score_gauge(self, current_score: float, 
                                 previous_score: Optional[float] = None) -> go.Figure:
        
        if current_score >= 750:
            color = self.color_scheme['excellent']
            level = '优秀'
        elif current_score >= 700:
            color = self.color_scheme['good']
            level = '良好'
        elif current_score >= 650:
            color = self.color_scheme['fair']
            level = '一般'
        else:
            color = self.color_scheme['poor']
            level = '需提升'
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"信用评分 - {level}", 'font': {'size': 24}},
            delta={
                'reference': previous_score if previous_score else current_score,
                'increasing': {'color': self.color_scheme['excellent']},
                'decreasing': {'color': self.color_scheme['poor']}
            },
            number={'suffix': '分', 'font': {'size': 48}},
            gauge={
                'axis': {
                    'range': [300, 850],
                    'tickwidth': 1,
                    'tickcolor': self.color_scheme['grid']
                },
                'bar': {'color': color, 'thickness': 0.75},
                'bgcolor': 'white',
                'borderwidth': 2,
                'bordercolor': self.color_scheme['grid'],
                'steps': [
                    {'range': [300, 600], 'color': '#FEE2E2'},
                    {'range': [600, 650], 'color': '#FEF3C7'},
                    {'range': [650, 700], 'color': '#DBEAFE'},
                    {'range': [700, 750], 'color': '#BBF7D0'},
                    {'range': [750, 850], 'color': '#D1FAE5'}
                ],
                'threshold': {
                    'line': {'color': color, 'width': 4},
                    'thickness': 0.75,
                    'value': current_score
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            **self.layout_template
        )
        
        return fig
    
    def create_factor_radar_chart(self, factors: Dict[str, float]) -> go.Figure:
        
        categories = list(factors.keys())
        values = list(factors.values())
        
        normalized_values = [(v - min(values)) / (max(values) - min(values)) * 100 
                            if max(values) > min(values) else 50 
                            for v in values]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_values + [normalized_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.2)',
            line=dict(color='rgba(59, 130, 246, 1)', width=2),
            name='当前表现'
        ))
        
        optimal_values = [80] * (len(categories) + 1)
        fig.add_trace(go.Scatterpolar(
            r=optimal_values,
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(16, 185, 129, 0.1)',
            line=dict(color='rgba(16, 185, 129, 0.5)', width=1, dash='dash'),
            name='理想水平'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    showticklabels=True,
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=12)
                )
            ),
            title={
                'text': '信用健康五维度分析',
                'font': {'size': 20},
                'x': 0.5,
                'xanchor': 'center'
            },
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            ),
            **self.layout_template
        )
        
        return fig
    
    def create_score_trend_chart(self, historical_data: List[Dict]) -> go.Figure:
        
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['score'],
            mode='lines+markers',
            name='信用评分',
            line=dict(color=self.color_scheme['good'], width=3),
            marker=dict(size=8, color=self.color_scheme['good']),
            fill='tonexty',
            fillcolor='rgba(59, 130, 246, 0.1)'
        ))
        
        if 'predicted_score' in df.columns:
            future_df = df[df['predicted_score'].notna()]
            fig.add_trace(go.Scatter(
                x=future_df['date'],
                y=future_df['predicted_score'],
                mode='lines+markers',
                name='预测评分',
                line=dict(color=self.color_scheme['fair'], width=2, dash='dash'),
                marker=dict(size=6, color=self.color_scheme['fair'])
            ))
        
        threshold_scores = [
            (750, '优秀', self.color_scheme['excellent']),
            (700, '良好', self.color_scheme['good']),
            (650, '一般', self.color_scheme['fair'])
        ]
        
        for score, label, color in threshold_scores:
            fig.add_hline(
                y=score,
                line_dash="dot",
                line_color=color,
                opacity=0.3,
                annotation_text=label,
                annotation_position="right"
            )
        
        fig.update_layout(
            title={
                'text': '信用评分趋势',
                'font': {'size': 20},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='日期',
            yaxis_title='信用评分',
            hovermode='x unified',
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5
            ),
            **self.layout_template
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=self.color_scheme['grid'])
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=self.color_scheme['grid'])
        
        return fig
    
    def create_utilization_breakdown_chart(self, accounts: List[Dict]) -> go.Figure:
        
        df = pd.DataFrame(accounts)
        
        df['utilization_pct'] = (df['used_limit'] / df['total_limit'] * 100).round(1)
        df = df.sort_values('utilization_pct', ascending=False)
        
        colors = []
        for util in df['utilization_pct']:
            if util >= 70:
                colors.append(self.color_scheme['poor'])
            elif util >= 50:
                colors.append(self.color_scheme['fair'])
            elif util >= 30:
                colors.append(self.color_scheme['good'])
            else:
                colors.append(self.color_scheme['excellent'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df['account_name'],
            y=df['utilization_pct'],
            marker_color=colors,
            text=df['utilization_pct'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>' +
                         '使用率: %{y:.1f}%<br>' +
                         '<extra></extra>'
        ))
        
        fig.add_hline(
            y=30,
            line_dash="dash",
            line_color=self.color_scheme['excellent'],
            annotation_text="理想线(30%)",
            annotation_position="right"
        )
        
        fig.add_hline(
            y=70,
            line_dash="dash",
            line_color=self.color_scheme['poor'],
            annotation_text="警戒线(70%)",
            annotation_position="right"
        )
        
        fig.update_layout(
            title={
                'text': '各账户信用使用率',
                'font': {'size': 20},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='账户',
            yaxis_title='使用率 (%)',
            height=400,
            showlegend=False,
            **self.layout_template
        )
        
        fig.update_yaxes(range=[0, 100])
        
        return fig
    
    def create_payment_history_calendar(self, payment_data: List[Dict], 
                                       months: int = 12) -> go.Figure:
        
        df = pd.DataFrame(payment_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= datetime.now() - timedelta(days=months*30)]
        
        df['week'] = df['date'].dt.isocalendar().week
        df['weekday'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.strftime('%Y-%m')
        
        status_colors = {
            'on_time': self.color_scheme['excellent'],
            'late': self.color_scheme['poor'],
            'early': self.color_scheme['good'],
            'no_payment': '#E5E7EB'
        }
        
        df['color'] = df['status'].map(status_colors)
        
        fig = go.Figure()
        
        for month in df['month'].unique():
            month_data = df[df['month'] == month]
            
            fig.add_trace(go.Scatter(
                x=month_data['weekday'],
                y=month_data['week'],
                mode='markers',
                marker=dict(
                    size=15,
                    color=month_data['color'],
                    symbol='square',
                    line=dict(width=1, color='white')
                ),
                name=month,
                hovertemplate='<b>%{text}</b><br>' +
                             '状态: %{customdata}<br>' +
                             '<extra></extra>',
                text=month_data['date'].dt.strftime('%Y-%m-%d'),
                customdata=month_data['status']
            ))
        
        fig.update_layout(
            title={
                'text': '还款历史日历',
                'font': {'size': 20},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis=dict(
                tickmode='array',
                tickvals=[0, 1, 2, 3, 4, 5, 6],
                ticktext=['周一', '周二', '周三', '周四', '周五', '周六', '周日'],
                title=''
            ),
            yaxis=dict(title='周数'),
            height=500,
            showlegend=False,
            **self.layout_template
        )
        
        return fig
    
    def create_impact_simulation_chart(self, scenarios: List[Dict]) -> go.Figure:
        
        scenario_names = [s['name'] for s in scenarios]
        immediate_impact = [s['immediate_impact'] for s in scenarios]
        month_3_impact = [s['month_3_impact'] for s in scenarios]
        month_6_impact = [s['month_6_impact'] for s in scenarios]
        month_12_impact = [s['month_12_impact'] for s in scenarios]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='即时影响',
            x=scenario_names,
            y=immediate_impact,
            marker_color=self.color_scheme['poor']
        ))
        
        fig.add_trace(go.Bar(
            name='3个月后',
            x=scenario_names,
            y=month_3_impact,
            marker_color=self.color_scheme['fair']
        ))
        
        fig.add_trace(go.Bar(
            name='6个月后',
            x=scenario_names,
            y=month_6_impact,
            marker_color=self.color_scheme['good']
        ))
        
        fig.add_trace(go.Bar(
            name='12个月后',
            x=scenario_names,
            y=month_12_impact,
            marker_color=self.color_scheme['excellent']
        ))
        
        fig.update_layout(
            title={
                'text': '不同行动的影响模拟',
                'font': {'size': 20},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='行动场景',
            yaxis_title='信用评分变化',
            barmode='group',
            height=450,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            **self.layout_template
        )
        
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
        
        return fig
    
    def create_debt_composition_sunburst(self, debt_data: Dict) -> go.Figure:
        
        labels = ['总债务']
        parents = ['']
        values = [debt_data['total']]
        colors = []
        
        categories = {
            '抵押贷款': debt_data.get('mortgage', 0),
            '汽车贷款': debt_data.get('auto_loan', 0),
            '信用卡': debt_data.get('credit_card', 0),
            '学生贷款': debt_data.get('student_loan', 0),
            '个人贷款': debt_data.get('personal_loan', 0)
        }
        
        for category, amount in categories.items():
            if amount > 0:
                labels.append(category)
                parents.append('总债务')
                values.append(amount)
        
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(
                colors=['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899'],
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>%{label}</b><br>' +
                         '金额: ¥%{value:,.0f}<br>' +
                         '占比: %{percentParent}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': '债务构成分析',
                'font': {'size': 20},
                'x': 0.5,
                'xanchor': 'center'
            },
            height=500,
            **self.layout_template
        )
        
        return fig
    
    def create_improvement_progress_chart(self, milestones: List[Dict], 
                                         current_progress: Dict) -> go.Figure:
        
        milestone_names = [m['description'] for m in milestones]
        target_months = [m['month'] for m in milestones]
        completed = [m.get('completed', False) for m in milestones]
        
        colors = [self.color_scheme['excellent'] if c else self.color_scheme['grid'] 
                 for c in completed]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=target_months,
            y=[1] * len(milestones),
            orientation='h',
            marker=dict(color=colors),
            text=milestone_names,
            textposition='inside',
            insidetextanchor='middle',
            hovertemplate='<b>%{text}</b><br>' +
                         '目标月份: 第%{x}个月<br>' +
                         '状态: %{customdata}<br>' +
                         '<extra></extra>',
            customdata=['已完成' if c else '进行中' for c in completed]
        ))
        
        current_month = current_progress.get('current_month', 0)
        fig.add_vline(
            x=current_month,
            line_dash="dash",
            line_color=self.color_scheme['good'],
            line_width=3,
            annotation_text=f"当前进度 (第{current_month}月)",
            annotation_position="top"
        )
        
        fig.update_layout(
            title={
                'text': '提升计划进度跟踪',
                'font': {'size': 20},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='时间轴 (月)',
            yaxis=dict(visible=False),
            height=300,
            showlegend=False,
            **self.layout_template
        )
        
        return fig
    
    def create_factor_contribution_waterfall(self, contributions: Dict) -> go.Figure:
        
        sorted_contributions = sorted(contributions.items(), 
                                     key=lambda x: abs(x[1]), 
                                     reverse=True)
        
        factors = [item[0] for item in sorted_contributions]
        values = [item[1] for item in sorted_contributions]
        
        base_score = 500
        
        fig = go.Figure(go.Waterfall(
            name="评分构成",
            orientation="v",
            measure=["relative"] * len(factors) + ["total"],
            x=factors + ["总分"],
            textposition="outside",
            text=[f"{v:+.0f}" for v in values] + [f"{base_score + sum(values):.0f}"],
            y=values + [base_score + sum(values)],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": self.color_scheme['poor']}},
            increasing={"marker": {"color": self.color_scheme['excellent']}},
            totals={"marker": {"color": self.color_scheme['good']}}
        ))
        
        fig.update_layout(
            title={
                'text': '信用评分构成瀑布图',
                'font': {'size': 20},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='影响因素',
            yaxis_title='评分',
            height=500,
            showlegend=False,
            **self.layout_template
        )
        
        return fig
    
    def create_comprehensive_dashboard(self, user_data: Dict) -> Dict[str, go.Figure]:
        
        dashboard = {}
        
        dashboard['score_gauge'] = self.create_credit_score_gauge(
            current_score=user_data.get('current_score', 700),
            previous_score=user_data.get('previous_score')
        )
        
        if 'factor_scores' in user_data:
            dashboard['radar_chart'] = self.create_factor_radar_chart(
                user_data['factor_scores']
            )
        
        if 'historical_scores' in user_data:
            dashboard['trend_chart'] = self.create_score_trend_chart(
                user_data['historical_scores']
            )
        
        if 'accounts' in user_data:
            dashboard['utilization_chart'] = self.create_utilization_breakdown_chart(
                user_data['accounts']
            )
        
        if 'debt_composition' in user_data:
            dashboard['debt_sunburst'] = self.create_debt_composition_sunburst(
                user_data['debt_composition']
            )
        
        if 'factor_contributions' in user_data:
            dashboard['waterfall_chart'] = self.create_factor_contribution_waterfall(
                user_data['factor_contributions']
            )
        
        return dashboard
    
    def export_dashboard_html(self, dashboard: Dict[str, go.Figure], 
                            filename: str = 'credit_dashboard.html'):
        
        html_parts = ['<html><head><title>信用健康仪表盘</title></head><body>']
        html_parts.append('<h1 style="text-align: center;">信用健康管理仪表盘</h1>')
        
        for title, fig in dashboard.items():
            html_parts.append(f'<div id="{title}">')
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            html_parts.append('</div>')
        
        html_parts.append('</body></html>')
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_parts))
        
        print(f"Dashboard exported to {filename}")