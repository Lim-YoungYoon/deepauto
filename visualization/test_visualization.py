#!/usr/bin/env python3
"""
Agent Visualization Test Script

에이전트 토폴로지 및 상호작용 시각화 모듈을 테스트합니다.
"""

import sys
import os
import time
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualization.visualization_adapter import test_visualization_adapter, get_visualization_adapter
from visualization.visualization import AgentVisualizationManager


def test_basic_visualization():
    """기본 시각화 기능 테스트"""
    print("Testing Basic Visualization Features...")
    
    # 시각화 어댑터 테스트
    adapter = test_visualization_adapter()
    
    # 시각화 매니저 가져오기
    viz_manager = adapter.get_visualization_manager()
    
    print("\nBasic visualization test completed!")
    return adapter


def test_api_endpoints():
    """API 엔드포인트 테스트"""
    print("\nTesting API Endpoints...")
    
    # FastAPI 앱을 임포트하여 테스트
    try:
        from api.visualization_routes import visualization_router
        print("Visualization router imported successfully")
        
        # 라우터 정보 출력
        print(f"Available routes:")
        for route in visualization_router.routes:
            print(f"   {route.methods} {route.path}")
            
    except ImportError as e:
        print(f"Failed to import visualization router: {e}")
        return False
    
    print("API endpoints test completed!")
    return True


def test_visualization_manager():
    """시각화 매니저 직접 테스트"""
    print("\nTesting Visualization Manager Directly...")
    
    try:
        # 시각화 매니저 초기화
        viz_manager = AgentVisualizationManager()
        
        # 시뮬레이션 데이터 생성
        viz_manager.simulate_interactions(15)
        
        # 상태 요약 출력
        summary = viz_manager.visualizer.get_agent_status_summary()
        print(f"Agent Status Summary:")
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        # 에이전트 목록 출력
        print(f"\nAvailable Agents:")
        for agent_id, agent in viz_manager.visualizer.agents.items():
            print(f"   - {agent.name} ({agent.agent_type}): {agent.status}")
        
        print("Visualization manager test completed!")
        return True
        
    except Exception as e:
        print(f"Visualization manager test failed: {e}")
        return False


def test_chart_generation():
    """차트 생성 테스트"""
    print("\nTesting Chart Generation...")
    
    try:
        viz_manager = AgentVisualizationManager()
        viz_manager.simulate_interactions(10)
        
        # 토폴로지 그래프 생성
        print("   Generating topology graph...")
        topology_fig = viz_manager.visualizer.create_topology_graph()
        print("   Topology graph created")
        
        # 타임라인 생성
        print("   Generating timeline...")
        timeline_fig = viz_manager.visualizer.create_interaction_timeline()
        print("   Timeline created")
        
        # 대시보드 생성
        print("   Generating dashboard...")
        dashboard_fig = viz_manager.visualizer.create_performance_dashboard()
        print("   Dashboard created")
        
        # 에이전트 상세 정보 생성
        print("   Generating agent details...")
        details_fig = viz_manager.visualizer.create_agent_details_view("decision_agent")
        print("   Agent details created")
        
        # 차트 저장 (선택사항)
        save_charts = input("\nSave charts as PNG files? (y/n): ").lower().strip()
        if save_charts == 'y':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            topology_fig.savefig(f"test_topology_{timestamp}.png", dpi=150, bbox_inches='tight')
            timeline_fig.savefig(f"test_timeline_{timestamp}.png", dpi=150, bbox_inches='tight')
            dashboard_fig.savefig(f"test_dashboard_{timestamp}.png", dpi=150, bbox_inches='tight')
            details_fig.savefig(f"test_agent_details_{timestamp}.png", dpi=150, bbox_inches='tight')
            print("   Charts saved successfully!")
        
        print("Chart generation test completed!")
        return True
        
    except Exception as e:
        print(f"Chart generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_export():
    """데이터 내보내기 테스트"""
    print("\nTesting Data Export...")
    
    try:
        adapter = get_visualization_adapter()
        adapter.start_monitoring()
        
        # 시뮬레이션 데이터 생성
        adapter.viz_manager.simulate_interactions(5)
        
        # 데이터 내보내기
        filename = f"test_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        adapter.export_current_state(filename)
        
        # 파일 존재 확인
        if os.path.exists(filename):
            print(f"   Data exported to {filename}")
            
            # 파일 크기 확인
            file_size = os.path.getsize(filename)
            print(f"   File size: {file_size} bytes")
            
            # 파일 삭제 (선택사항)
            delete_file = input(f"Delete test file {filename}? (y/n): ").lower().strip()
            if delete_file == 'y':
                os.remove(filename)
                print(f"   File {filename} deleted")
        else:
            print(f"   Failed to create file {filename}")
            return False
        
        print("Data export test completed!")
        return True
        
    except Exception as e:
        print(f"Data export test failed: {e}")
        return False


def main():
    """메인 테스트 함수"""
    print("Starting Agent Visualization Test Suite")
    print("=" * 50)
    
    test_results = []
    
    # 1. 기본 시각화 테스트
    try:
        adapter = test_basic_visualization()
        test_results.append(("Basic Visualization", True))
    except Exception as e:
        print(f"Basic visualization test failed: {e}")
        test_results.append(("Basic Visualization", False))
    
    # 2. API 엔드포인트 테스트
    try:
        result = test_api_endpoints()
        test_results.append(("API Endpoints", result))
    except Exception as e:
        print(f"API endpoints test failed: {e}")
        test_results.append(("API Endpoints", False))
    
    # 3. 시각화 매니저 테스트
    try:
        result = test_visualization_manager()
        test_results.append(("Visualization Manager", result))
    except Exception as e:
        print(f"Visualization manager test failed: {e}")
        test_results.append(("Visualization Manager", False))
    
    # 4. 차트 생성 테스트
    try:
        result = test_chart_generation()
        test_results.append(("Chart Generation", result))
    except Exception as e:
        print(f"Chart generation test failed: {e}")
        test_results.append(("Chart Generation", False))
    
    # 5. 데이터 내보내기 테스트
    try:
        result = test_data_export()
        test_results.append(("Data Export", result))
    except Exception as e:
        print(f"Data export test failed: {e}")
        test_results.append(("Data Export", False))
    
    # 테스트 결과 요약
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"Total: {total}, Passed: {passed}, Failed: {total - passed}")
    
    if passed == total:
        print("All tests passed! Visualization module is working correctly.")
    else:
        print("Some tests failed. Please check the error messages above.")
    
    print("\nAvailable API Endpoints:")
    print("   GET  /api/visualization/status")
    print("   GET  /api/visualization/topology")
    print("   GET  /api/visualization/timeline")
    print("   GET  /api/visualization/dashboard")
    print("   GET  /api/visualization/agent/{agent_name}")
    print("   GET  /api/visualization/interactions")
    print("   POST /api/visualization/export")
    print("   POST /api/visualization/monitoring/start")
    print("   POST /api/visualization/monitoring/stop")
    print("   POST /api/visualization/simulate")
    print("   GET  /api/visualization/dashboard/embed")
    
    print("\nTo start the server with visualization:")
    print("   python app.py")
    print("   Then visit: http://localhost:8000/api/visualization/dashboard/embed")


if __name__ == "__main__":
    main() 