#!/usr/bin/env python3
"""
Real Agent Activity Test Script

실제 에이전트 활동을 시각화하는 테스트 스크립트입니다.
"""

import sys
import os
import time
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multi_agents.decision import process_query
from visualization.visualization_adapter import get_visualization_adapter


def test_real_agent_activity():
    """실제 에이전트 활동 테스트"""
    print("Testing Real Agent Activity with Visualization...")
    print("=" * 60)
    
    # 시각화 어댑터 가져오기
    adapter = get_visualization_adapter()
    
    # 테스트 쿼리들
    test_queries = [
        "안녕하세요! 오늘 날씨는 어때요?",  # Conversation Agent
        "AI에 대해 설명해주세요",  # RAG Agent (문서 기반)
        "최신 AI 기술 동향은 어떻게 되나요?",  # Web Search Agent
        "감사합니다!",  # Conversation Agent
    ]
    
    print("Processing test queries...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)
        
        try:
            # 실제 에이전트 처리
            result = process_query(query)
            print(f"Query {i} processed successfully")
            
            # 잠시 대기 (시각화 업데이트를 위해)
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing query {i}: {e}")
    
    # 시각화 상태 확인
    print("\nVisualization Status:")
    print("-" * 40)
    
    summary = adapter.get_agent_status_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # 에이전트별 상세 정보
    print("\nAgent Details:")
    print("-" * 40)
    
    for agent_id, agent in adapter.get_visualizer().agents.items():
        print(f"   {agent.name}:")
        print(f"     - Status: {agent.status}")
        print(f"     - Last Activity: {agent.last_activity}")
        print(f"     - Performance: {agent.performance_metrics}")
    
    # 상호작용 기록
    print(f"\nRecent Interactions ({len(adapter.get_visualizer().interactions)} total):")
    print("-" * 40)
    
    recent_interactions = adapter.get_visualizer().interactions[-10:]  # 최근 10개
    for i, interaction in enumerate(recent_interactions, 1):
        print(f"   {i}. {interaction.source} → {interaction.target} ({interaction.interaction_type})")
        print(f"      Time: {interaction.timestamp.strftime('%H:%M:%S')}")
        print(f"      Duration: {interaction.duration:.2f}s")
    
    print("\nAvailable Visualizations:")
    print("-" * 40)
    print("   - Web Dashboard: http://localhost:8000/api/visualization/dashboard/embed")
    print("   - Topology Graph: http://localhost:8000/api/visualization/topology")
    print("   - Timeline: http://localhost:8000/api/visualization/timeline")
    print("   - Performance Dashboard: http://localhost:8000/api/visualization/dashboard")
    
    return adapter


def test_with_pdf_upload():
    """PDF 업로드 시나리오 테스트"""
    print("\nTesting PDF Upload Scenario...")
    print("=" * 60)
    
    # PDF 업로드 시뮬레이션
    pdf_query = {
        "text": "이 문서에서 AI에 대해 설명해주세요",
        "pdf": "test_document.pdf"  # 실제로는 파일 경로
    }
    
    try:
        result = process_query(pdf_query)
        print("PDF upload scenario processed successfully")
        
        # RAG Agent 활동 확인
        adapter = get_visualization_adapter()
        rag_agent = adapter.get_visualizer().agents.get("rag_agent")
        if rag_agent:
            print(f"   RAG Agent Status: {rag_agent.status}")
            print(f"   RAG Agent Performance: {rag_agent.performance_metrics}")
        
    except Exception as e:
        print(f"Error in PDF upload scenario: {e}")


def main():
    """메인 테스트 함수"""
    print("Real Agent Activity Test Suite")
    print("=" * 60)
    
    try:
        # 1. 기본 에이전트 활동 테스트
        adapter = test_real_agent_activity()
        
        # 2. PDF 업로드 시나리오 테스트
        test_with_pdf_upload()
        
        print("\n" + "=" * 60)
        print("Real agent activity test completed!")
        print("\nNext Steps:")
        print("   1. Start the server: python app.py")
        print("   2. Open browser: http://localhost:8000/api/visualization/dashboard/embed")
        print("   3. Use the chat interface to see real-time agent activity")
        print("   4. Upload a PDF to see RAG agent in action")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 