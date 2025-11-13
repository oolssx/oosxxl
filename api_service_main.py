from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import asyncio
import uvicorn
from enum import Enum
import redis
import json


app = FastAPI(
    title="信用健康管家API",
    description="全方位信用管理智能服务",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UserRole(str, Enum):
    REGULAR = "regular"
    PREMIUM = "premium"
    VIP = "vip"


class CreditScoreRequest(BaseModel):
    user_id: str = Field(..., description="用户唯一标识")
    include_explanation: bool = Field(True, description="是否包含解释")
    include_breakdown: bool = Field(True, description="是否包含详细分解")


class CreditScoreResponse(BaseModel):
    user_id: str
    credit_score: float
    confidence: float
    score_level: str
    timestamp: datetime
    explanation: Optional[str]
    breakdown: Optional[Dict]
    component_scores: Optional[Dict]


class ActionMonitorRequest(BaseModel):
    user_id: str
    action_type: str
    action_data: Dict[str, Any]
    request_warning: bool = Field(True, description="是否需要预警")


class WarningResponse(BaseModel):
    warning_triggered: bool
    risk_level: str
    risk_score: float
    impact_prediction: Dict
    warning_message: Optional[str]
    recommendations: List[Dict]
    alternatives: List[Dict]


class ImprovementPlanRequest(BaseModel):
    user_id: str
    goal_type: str
    target_score: float
    timeline_months: int = Field(6, ge=1, le=24)


class ImprovementPlanResponse(BaseModel):
    plan_id: str
    user_id: str
    current_score: float
    target_score: float
    timeline_months: int
    actions: List[Dict]
    milestones: List[Dict]
    success_probability: float
    narrative_explanation: str


class SimulationRequest(BaseModel):
    user_id: str
    scenario_description: str
    feature_changes: Dict[str, Any]


class SimulationResponse(BaseModel):
    scenario_description: str
    current_score: float
    projected_score: float
    score_change: float
    time_projections: Dict[str, float]
    confidence: float
    explanation: str


class DashboardRequest(BaseModel):
    user_id: str
    dashboard_type: str = Field("comprehensive", description="仪表盘类型")
    time_range: str = Field("12_months", description="时间范围")


redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)


def get_current_user(user_id: str):
    user_key = f"user:{user_id}"
    user_data = redis_client.hgetall(user_key)
    
    if not user_data:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    return user_data


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>信用健康管家API</title>
            <style>
                body { font-family: system-ui; max-width: 800px; margin: 50px auto; padding: 20px; }
                h1 { color: #3B82F6; }
                .endpoint { background: #F3F4F6; padding: 15px; margin: 10px 0; border-radius: 8px; }
                .method { color: #10B981; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>🛡️ 信用健康管家API服务</h1>
            <p>欢迎使用信用健康管家API。访问 <a href="/docs">/docs</a> 查看完整API文档。</p>
            
            <h2>核心功能</h2>
            <div class="endpoint">
                <span class="method">POST</span> /api/v1/credit-score/predict
                <p>获取用户信用评分预测</p>
            </div>
            <div class="endpoint">
                <span class="method">POST</span> /api/v1/monitoring/action
                <p>监控用户行为并触发预警</p>
            </div>
            <div class="endpoint">
                <span class="method">POST</span> /api/v1/improvement/plan
                <p>生成个性化信用提升计划</p>
            </div>
            <div class="endpoint">
                <span class="method">POST</span> /api/v1/simulation/scenario
                <p>模拟不同行为对信用的影响</p>
            </div>
        </body>
    </html>
    """


@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "credit-health-manager",
        "version": "1.0.0"
    }


@app.post("/api/v1/credit-score/predict", response_model=CreditScoreResponse)
async def predict_credit_score(request: CreditScoreRequest):
    try:
        user_data = get_current_user(request.user_id)
        
        features = {
            'credit_history_months': int(user_data.get('credit_history_months', 0)),
            'total_accounts': int(user_data.get('total_accounts', 0)),
            'credit_utilization': float(user_data.get('credit_utilization', 0)),
            'total_overdue_count': int(user_data.get('total_overdue_count', 0)),
            'hard_inquiries_6m': int(user_data.get('hard_inquiries_6m', 0)),
            'debt_to_income_ratio': float(user_data.get('debt_to_income_ratio', 0))
        }
        
        history_key = f"user:{request.user_id}:history"
        history_data = redis_client.lrange(history_key, 0, 23)
        
        prediction = {
            'score': 720.0,
            'confidence': 0.85,
            'component_scores': {
                'gbdt': 725.0,
                'lstm': 718.0,
                'rf': 717.0
            }
        }
        
        score_level = 'good'
        if prediction['score'] >= 750:
            score_level = 'excellent'
        elif prediction['score'] >= 700:
            score_level = 'good'
        elif prediction['score'] >= 650:
            score_level = 'fair'
        else:
            score_level = 'poor'
        
        explanation = None
        if request.include_explanation:
            explanation = f"您的信用评分为{prediction['score']:.0f}分，处于{score_level}水平。主要优势是信用历史长度和良好的还款记录。建议降低信用卡使用率以进一步提升评分。"
        
        breakdown = None
        if request.include_breakdown:
            breakdown = {
                'payment_history': {'score': 95, 'weight': 0.35, 'contribution': 33.25},
                'credit_utilization': {'score': 70, 'weight': 0.30, 'contribution': 21.0},
                'credit_history': {'score': 85, 'weight': 0.15, 'contribution': 12.75},
                'credit_mix': {'score': 80, 'weight': 0.10, 'contribution': 8.0},
                'new_credit': {'score': 75, 'weight': 0.10, 'contribution': 7.5}
            }
        
        return CreditScoreResponse(
            user_id=request.user_id,
            credit_score=prediction['score'],
            confidence=prediction['confidence'],
            score_level=score_level,
            timestamp=datetime.now(),
            explanation=explanation,
            breakdown=breakdown,
            component_scores=prediction['component_scores']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.post("/api/v1/monitoring/action", response_model=WarningResponse)
async def monitor_user_action(request: ActionMonitorRequest, 
                             background_tasks: BackgroundTasks):
    try:
        user_data = get_current_user(request.user_id)
        
        risk_assessment = {
            'risk_score': 0.65,
            'risk_level': 'high'
        }
        
        impact_prediction = {
            'immediate_impact': -15.0,
            'current_score': float(user_data.get('current_score', 700)),
            'projected_score': float(user_data.get('current_score', 700)) - 15.0,
            'time_projections': {
                '1_month': -15.0,
                '3_months': -10.0,
                '6_months': -5.0,
                '12_months': 0.0
            },
            'recovery_time_estimate': 6
        }
        
        warning_message = None
        recommendations = []
        alternatives = []
        
        if risk_assessment['risk_score'] > 0.4:
            warning_message = f"检测到高风险操作！本次{request.action_type}预计会降低您的信用评分约{abs(impact_prediction['immediate_impact']):.0f}分。"
            
            recommendations = [
                {
                    'priority': 'high',
                    'text': '建议推迟此操作直到完成其他重要信用活动',
                    'reason': '避免短期内信用评分大幅波动'
                },
                {
                    'priority': 'medium',
                    'text': '如确需进行，建议先降低信用卡使用率',
                    'reason': '可部分抵消负面影响'
                }
            ]
            
            alternatives = [
                {
                    'type': '信用卡分期',
                    'description': '使用现有信用卡额度进行分期',
                    'pros': ['不增加征信查询', '审批快'],
                    'cons': ['可能费率稍高']
                }
            ]
            
            background_tasks.add_task(log_warning, request.user_id, request.action_type, risk_assessment)
        
        return WarningResponse(
            warning_triggered=risk_assessment['risk_score'] > 0.4,
            risk_level=risk_assessment['risk_level'],
            risk_score=risk_assessment['risk_score'],
            impact_prediction=impact_prediction,
            warning_message=warning_message,
            recommendations=recommendations,
            alternatives=alternatives
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"监控失败: {str(e)}")


@app.post("/api/v1/improvement/plan", response_model=ImprovementPlanResponse)
async def generate_improvement_plan(request: ImprovementPlanRequest):
    try:
        user_data = get_current_user(request.user_id)
        
        current_score = float(user_data.get('current_score', 680))
        
        actions = [
            {
                'action_id': 'reduce_utilization',
                'title': '降低信用卡使用率',
                'priority': 'critical',
                'estimated_impact': 25.0,
                'time_to_impact': 1,
                'effort_level': 'medium'
            },
            {
                'action_id': 'setup_autopay',
                'title': '设置自动还款',
                'priority': 'high',
                'estimated_impact': 15.0,
                'time_to_impact': 1,
                'effort_level': 'easy'
            },
            {
                'action_id': 'pay_down_debt',
                'title': '偿还高息债务',
                'priority': 'high',
                'estimated_impact': 18.0,
                'time_to_impact': 3,
                'effort_level': 'hard'
            }
        ]
        
        milestones = []
        score_increment = (request.target_score - current_score) / request.timeline_months
        
        for month in [1, 3, 6]:
            if month <= request.timeline_months:
                milestones.append({
                    'month': month,
                    'target_score': current_score + (score_increment * month),
                    'description': f'第{month}个月目标',
                    'deadline': (datetime.now() + timedelta(days=30*month)).strftime('%Y-%m-%d')
                })
        
        success_probability = min(0.85, 0.5 + (request.timeline_months / 24))
        
        narrative = f"根据您的目标，我们为您制定了{request.timeline_months}个月的提升计划。通过执行{len(actions)}个关键行动，预计可以将您的信用评分从{current_score:.0f}分提升至{request.target_score:.0f}分。成功概率约为{success_probability*100:.0f}%。"
        
        plan_id = f"plan_{request.user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        redis_client.hset(f"plan:{plan_id}", mapping={
            'user_id': request.user_id,
            'created_at': datetime.now().isoformat(),
            'target_score': request.target_score,
            'timeline_months': request.timeline_months
        })
        
        return ImprovementPlanResponse(
            plan_id=plan_id,
            user_id=request.user_id,
            current_score=current_score,
            target_score=request.target_score,
            timeline_months=request.timeline_months,
            actions=actions,
            milestones=milestones,
            success_probability=success_probability,
            narrative_explanation=narrative
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"计划生成失败: {str(e)}")


@app.post("/api/v1/simulation/scenario", response_model=SimulationResponse)
async def simulate_scenario(request: SimulationRequest):
    try:
        user_data = get_current_user(request.user_id)
        
        current_score = float(user_data.get('current_score', 700))
        
        total_change = 0
        for feature, change in request.feature_changes.items():
            if feature == 'credit_utilization':
                if isinstance(change, dict):
                    delta = change['value'] - float(user_data.get('credit_utilization', 0))
                    total_change -= delta * 50
            elif feature == 'hard_inquiries_6m':
                total_change -= change.get('value', 0) * 5
            elif feature == 'total_debt':
                if isinstance(change, dict):
                    delta = change['value'] - float(user_data.get('total_debt', 0))
                    total_change -= (delta / 10000) * 2
        
        projected_score = current_score + total_change
        
        time_projections = {
            '1_month': total_change,
            '3_months': total_change * 0.8,
            '6_months': total_change * 0.6,
            '12_months': total_change * 0.3
        }
        
        explanation = f"如果{request.scenario_description}，您的信用评分预计会{('提升' if total_change > 0 else '降低')}{abs(total_change):.0f}分。"
        
        if total_change < -10:
            explanation += f"这是一个较大的负面影响，建议谨慎考虑。预计需要{6 if abs(total_change) < 20 else 12}个月才能完全恢复。"
        elif total_change > 10:
            explanation += "这是一个积极的改变，建议尽快实施。"
        
        return SimulationResponse(
            scenario_description=request.scenario_description,
            current_score=current_score,
            projected_score=projected_score,
            score_change=total_change,
            time_projections=time_projections,
            confidence=0.75,
            explanation=explanation
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模拟失败: {str(e)}")


@app.get("/api/v1/dashboard/{user_id}")
async def get_user_dashboard(user_id: str, dashboard_type: str = "comprehensive"):
    try:
        user_data = get_current_user(user_id)
        
        dashboard_data = {
            'user_id': user_id,
            'dashboard_type': dashboard_type,
            'generated_at': datetime.now().isoformat(),
            'credit_score': {
                'current': float(user_data.get('current_score', 700)),
                'previous': float(user_data.get('previous_score', 695)),
                'change': float(user_data.get('current_score', 700)) - float(user_data.get('previous_score', 695))
            },
            'factor_scores': {
                '支付历史': 90,
                '信用使用': 70,
                '信用历史': 85,
                '信用组合': 75,
                '新开账户': 80
            },
            'alerts': {
                'active_warnings': 2,
                'pending_actions': 3,
                'milestones_achieved': 1
            },
            'recommendations': [
                {'type': 'urgent', 'message': '信用卡使用率偏高，建议降至30%以下'},
                {'type': 'info', 'message': '恭喜！连续6个月按时还款'}
            ]
        }
        
        return JSONResponse(content=dashboard_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"仪表盘加载失败: {str(e)}")


@app.get("/api/v1/user/{user_id}/history")
async def get_user_history(user_id: str, days: int = 365):
    try:
        history_key = f"user:{user_id}:score_history"
        
        history_data = []
        for i in range(min(days, 365)):
            date = datetime.now() - timedelta(days=i)
            score = 700 + (i % 50) - 25
            
            history_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'score': score,
                'change': 0
            })
        
        history_data.reverse()
        
        return JSONResponse(content={
            'user_id': user_id,
            'history': history_data,
            'count': len(history_data)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"历史数据获取失败: {str(e)}")


@app.get("/api/v1/factors/explain/{factor_name}")
async def explain_factor(factor_name: str):
    factor_explanations = {
        'payment_history': {
            'name': '支付历史',
            'weight': 0.35,
            'description': '按时还款的历史记录，是信用评分中最重要的因素',
            'optimization_tips': [
                '设置自动还款避免遗忘',
                '保持连续按时还款记录',
                '如有逾期尽快还清'
            ],
            'common_mistakes': [
                '只还最低还款额导致利息累积',
                '忘记还款日期',
                '多张卡难以管理'
            ]
        },
        'credit_utilization': {
            'name': '信用使用率',
            'weight': 0.30,
            'description': '已用信用额度占总额度的比例',
            'optimal_range': [0.1, 0.3],
            'optimization_tips': [
                '将使用率保持在30%以下',
                '账单日前分批还款',
                '考虑申请提额降低使用率'
            ]
        }
    }
    
    if factor_name not in factor_explanations:
        raise HTTPException(status_code=404, detail="因素不存在")
    
    return JSONResponse(content=factor_explanations[factor_name])


@app.post("/api/v1/user/{user_id}/authorize")
async def authorize_data_source(user_id: str, data_source: str, duration_days: int = 90):
    try:
        auth_key = f"auth:{user_id}:{data_source}"
        
        auth_data = {
            'user_id': user_id,
            'data_source': data_source,
            'authorized_at': datetime.now().isoformat(),
            'expiry': (datetime.now() + timedelta(days=duration_days)).isoformat(),
            'duration_days': duration_days
        }
        
        redis_client.setex(
            auth_key,
            duration_days * 86400,
            json.dumps(auth_data)
        )
        
        return JSONResponse(content={
            'success': True,
            'message': f'已授权访问{data_source}数据源',
            'expiry_date': auth_data['expiry']
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"授权失败: {str(e)}")


async def log_warning(user_id: str, action_type: str, risk_assessment: Dict):
    log_key = f"warning_log:{user_id}"
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'action_type': action_type,
        'risk_level': risk_assessment['risk_level'],
        'risk_score': risk_assessment['risk_score']
    }
    
    redis_client.lpush(log_key, json.dumps(log_entry))
    redis_client.ltrim(log_key, 0, 99)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "api_service_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )