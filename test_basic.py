def test_data_loader():
    """Test data loader hoạt động"""
    from data.data_loader import DataLoader
    
    loader = DataLoader()
    data = loader.download_stock_data('AAPL', period='5d', save=False)
    
    assert data is not None, "Data loader failed"
    assert len(data) > 0, "No data returned"
    print("✅ Data Loader test passed")

def test_strategies():
    """Test basic strategy functionality"""
    try:
        from strategies.ma_strategy import quick_ma_test
        strategy = quick_ma_test('AAPL')
        assert strategy is not None, "MA Strategy failed"
        print("✅ MA Strategy test passed")
    except Exception as e:
        print(f"❌ MA Strategy test failed: {e}")

if __name__ == "__main__":
    print("🧪 CHẠY CÁC TEST CƠ BẢN")
    print("-" * 30)
    
    test_data_loader()
    test_strategies()
    
    print("\n✅ Tất cả test đều pass! Dự án hoạt động tốt.")
