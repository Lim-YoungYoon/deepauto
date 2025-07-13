"""
Visualization Routes for Agent Topology

에이전트 토폴로지 및 상호작용 시각화를 위한 FastAPI 라우트입니다.
"""

import json
import base64
import io
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
import matplotlib
matplotlib.use('Agg')  # 백엔드에서 사용
import matplotlib.pyplot as plt
import seaborn as sns
from visualization.visualization_adapter import get_visualization_adapter

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

visualization_router = APIRouter(prefix="/api/visualization", tags=["visualization"])


def figure_to_base64(fig):
    """matplotlib figure를 base64 인코딩된 이미지로 변환"""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=150)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()


@visualization_router.get("/status")
async def get_agent_status():
    """에이전트 상태 요약 반환"""
    adapter = get_visualization_adapter()
    summary = adapter.get_agent_status_summary()
    
    # 각 에이전트의 상세 상태 추가
    agent_details = {}
    for agent_id, agent in adapter.get_visualizer().agents.items():
        agent_details[agent_id] = {
            "name": agent.name,
            "type": agent.agent_type,
            "status": agent.status,
            "last_activity": agent.last_activity.isoformat() if agent.last_activity else None,
            "performance_metrics": agent.performance_metrics
        }
    
    return {
        "summary": summary,
        "agents": agent_details,
        "timestamp": datetime.now().isoformat()
    }


@visualization_router.get("/topology")
async def get_topology_graph():
    """에이전트 토폴로지 그래프 생성"""
    adapter = get_visualization_adapter()
    viz_manager = adapter.get_visualization_manager()
    
    try:
        fig = viz_manager.visualizer.create_topology_graph()
        img_base64 = figure_to_base64(fig)
        plt.close(fig)
        
        return {
            "success": True,
            "image": img_base64,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@visualization_router.get("/timeline")
async def get_interaction_timeline():
    """상호작용 타임라인 생성"""
    adapter = get_visualization_adapter()
    viz_manager = adapter.get_visualization_manager()
    
    try:
        fig = viz_manager.visualizer.create_interaction_timeline()
        img_base64 = figure_to_base64(fig)
        plt.close(fig)
        
        return {
            "success": True,
            "image": img_base64,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@visualization_router.get("/dashboard")
async def get_performance_dashboard():
    """성능 대시보드 생성"""
    adapter = get_visualization_adapter()
    viz_manager = adapter.get_visualization_manager()
    
    try:
        fig = viz_manager.visualizer.create_performance_dashboard()
        img_base64 = figure_to_base64(fig)
        plt.close(fig)
        
        return {
            "success": True,
            "image": img_base64,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@visualization_router.get("/agent/{agent_name}")
async def get_agent_details(agent_name: str):
    """특정 에이전트 상세 정보"""
    adapter = get_visualization_adapter()
    viz_manager = adapter.get_visualization_manager()
    
    try:
        fig = viz_manager.visualizer.create_agent_details_view(agent_name)
        img_base64 = figure_to_base64(fig)
        plt.close(fig)
        
        # 에이전트 정보도 함께 반환
        agent = viz_manager.visualizer.agents.get(agent_name)
        agent_info = None
        if agent:
            agent_info = {
                "name": agent.name,
                "type": agent.agent_type,
                "description": agent.description,
                "capabilities": agent.capabilities,
                "dependencies": agent.dependencies,
                "status": agent.status,
                "last_activity": agent.last_activity.isoformat() if agent.last_activity else None,
                "performance_metrics": agent.performance_metrics
            }
        
        return {
            "success": True,
            "image": img_base64,
            "agent_info": agent_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@visualization_router.get("/interactions")
async def get_recent_interactions():
    """최근 상호작용 목록 반환"""
    adapter = get_visualization_adapter()
    visualizer = adapter.get_visualizer()
    
    # 최근 상호작용 필터링 (최근 50개)
    recent_interactions = visualizer.interactions[-50:]
    
    interactions_data = []
    for interaction in recent_interactions:
        interactions_data.append({
            "source": interaction.source,
            "target": interaction.target,
            "type": interaction.interaction_type,
            "timestamp": interaction.timestamp.isoformat(),
            "data_size": interaction.data_size,
            "duration": interaction.duration,
            "success": interaction.success
        })
    
    return {
        "interactions": interactions_data,
        "total_count": len(visualizer.interactions),
        "recent_count": len(interactions_data),
        "timestamp": datetime.now().isoformat()
    }


@visualization_router.post("/export")
async def export_visualization_data(request: Request):
    """시각화 데이터 내보내기"""
    adapter = get_visualization_adapter()
    
    try:
        body = await request.json()
        filename = body.get('filename', f'agent_topology_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        adapter.export_current_state(filename)
        
        return {
            "success": True,
            "filename": filename,
            "message": f"데이터가 {filename}에 저장되었습니다.",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@visualization_router.post("/monitoring/start")
async def start_monitoring():
    """모니터링 시작"""
    adapter = get_visualization_adapter()
    adapter.start_monitoring()
    
    return {
        "success": True,
        "message": "에이전트 모니터링이 시작되었습니다.",
        "timestamp": datetime.now().isoformat()
    }


@visualization_router.post("/monitoring/stop")
async def stop_monitoring():
    """모니터링 중지"""
    adapter = get_visualization_adapter()
    adapter.stop_monitoring()
    
    return {
        "success": True,
        "message": "에이전트 모니터링이 중지되었습니다.",
        "timestamp": datetime.now().isoformat()
    }


@visualization_router.post("/simulate")
async def simulate_interactions(request: Request):
    """상호작용 시뮬레이션"""
    adapter = get_visualization_adapter()
    
    try:
        body = await request.json()
        num_interactions = body.get('num_interactions', 10)
        adapter.viz_manager.simulate_interactions(num_interactions)
        
        return {
            "success": True,
            "message": f"{num_interactions}개의 상호작용이 시뮬레이션되었습니다.",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@visualization_router.get("/dashboard/embed", response_class=HTMLResponse)
async def get_embedded_dashboard():
    """임베드 가능한 대시보드 HTML 반환"""
    adapter = get_visualization_adapter()
    viz_manager = adapter.get_visualization_manager()
    
    try:
        # 여러 차트 생성
        topology_fig = viz_manager.visualizer.create_topology_graph(figsize=(8, 6))
        topology_img = figure_to_base64(topology_fig)
        plt.close(topology_fig)
        
        timeline_fig = viz_manager.visualizer.create_interaction_timeline(figsize=(10, 4))
        timeline_img = figure_to_base64(timeline_fig)
        plt.close(timeline_fig)
        
        dashboard_fig = viz_manager.visualizer.create_performance_dashboard(figsize=(12, 8))
        dashboard_img = figure_to_base64(dashboard_fig)
        plt.close(dashboard_fig)
        
        # 상태 요약
        summary = adapter.get_agent_status_summary()
        
        # HTML 템플릿 생성
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Agent Topology Dashboard</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .dashboard {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    text-align: center;
                }}
                .status-summary {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    padding: 20px;
                    background: #f8f9fa;
                }}
                .status-card {{
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .status-card h3 {{
                    margin: 0 0 10px 0;
                    color: #333;
                }}
                .status-card .value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #667eea;
                }}
                .charts {{
                    padding: 20px;
                }}
                .chart-section {{
                    margin-bottom: 30px;
                }}
                .chart-section h3 {{
                    color: #333;
                    border-bottom: 2px solid #667eea;
                    padding-bottom: 10px;
                    margin-bottom: 20px;
                }}
                .chart-container {{
                    text-align: center;
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .chart-container img {{
                    max-width: 100%;
                    height: auto;
                }}
                .timestamp {{
                    text-align: center;
                    color: #666;
                    font-size: 0.9em;
                    margin-top: 20px;
                    padding: 10px;
                    background: #f8f9fa;
                }}
            </style>
        </head>
        <body>
            <div class="dashboard">
                <div class="header">
                    <h1>🤖 Multi-Agent System Dashboard</h1>
                    <p>실시간 에이전트 토폴로지 및 성능 모니터링</p>
                </div>
                
                <div class="status-summary">
                    <div class="status-card">
                        <h3>총 에이전트</h3>
                        <div class="value">{summary['total_agents']}</div>
                    </div>
                    <div class="status-card">
                        <h3>활성 에이전트</h3>
                        <div class="value">{summary['active_agents']}</div>
                    </div>
                    <div class="status-card">
                        <h3>총 상호작용</h3>
                        <div class="value">{summary['total_interactions']}</div>
                    </div>
                    <div class="status-card">
                        <h3>최근 상호작용</h3>
                        <div class="value">{summary['recent_interactions']}</div>
                    </div>
                </div>
                
                <div class="charts">
                    <div class="chart-section">
                        <h3>📊 에이전트 토폴로지</h3>
                        <div class="chart-container">
                            <img src="data:image/png;base64,{topology_img}" alt="Agent Topology">
                        </div>
                    </div>
                    
                    <div class="chart-section">
                        <h3>📈 성능 대시보드</h3>
                        <div class="chart-container">
                            <img src="data:image/png;base64,{dashboard_img}" alt="Performance Dashboard">
                        </div>
                    </div>
                    
                    <div class="chart-section">
                        <h3>⏰ 상호작용 타임라인</h3>
                        <div class="chart-container">
                            <img src="data:image/png;base64,{timeline_img}" alt="Interaction Timeline">
                        </div>
                    </div>
                </div>
                
                <div class="timestamp">
                    마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 