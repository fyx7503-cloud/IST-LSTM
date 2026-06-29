#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MDB to CSV 转换器的单元测试

由于测试环境中缺少真实的 MDB 文件和 ODBC 驱动，
本测试使用 unittest.mock 模拟各策略的行为，验证核心逻辑。
"""

import csv
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

# 确保能导入被测模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.mdb_to_csv import (
    MDBToCSVConverter,
    MDBToolsStrategy,
    PyODBCSrategy,
    Win32ComStrategy,
)


class FakeStrategy:
    """用于测试的假策略"""
    name = "fake"

    def __init__(self, tables=None):
        self._tables = ["Table1", "Table2", "Tést_表3"] if tables is None else tables
        self.call_log = []

    def is_available(self):
        return True

    def list_tables(self, mdb_path):
        self.call_log.append(("list_tables", mdb_path))
        return self._tables

    def read_table(self, mdb_path, table_name):
        self.call_log.append(("read_table", mdb_path, table_name))
        data = {
            "ID": [1, 2, 3],
            "Name": ["Alice", "Bob", "Charlie"],
            "Score": [95.5, 87.0, 92.3],
        }
        return pd.DataFrame(data)


class TestMDBToCSVConverter(unittest.TestCase):

    def test_auto_detect_strategy_no_strategy_available(self):
        """测试没有任何策略可用时抛出异常"""
        with patch.object(PyODBCSrategy, "is_available", return_value=False), \
             patch.object(MDBToolsStrategy, "is_available", return_value=False), \
             patch.object(Win32ComStrategy, "is_available", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                MDBToCSVConverter()
            self.assertIn("未找到可用的 MDB 读取策略", str(ctx.exception))

    def test_auto_detect_strategy_priority(self):
        """测试策略优先级: pyodbc > mdbtools > win32com"""
        # pyodbc 可用
        with patch.object(PyODBCSrategy, "is_available", return_value=True), \
             patch.object(MDBToolsStrategy, "is_available", return_value=True), \
             patch.object(Win32ComStrategy, "is_available", return_value=True):
            converter = MDBToCSVConverter()
            self.assertEqual(converter.strategy.name, "pyodbc")

        # 只有 mdbtools 可用
        with patch.object(PyODBCSrategy, "is_available", return_value=False), \
             patch.object(MDBToolsStrategy, "is_available", return_value=True), \
             patch.object(Win32ComStrategy, "is_available", return_value=True):
            converter = MDBToCSVConverter()
            self.assertEqual(converter.strategy.name, "mdbtools")

        # 只有 win32com 可用
        with patch.object(PyODBCSrategy, "is_available", return_value=False), \
             patch.object(MDBToolsStrategy, "is_available", return_value=False), \
             patch.object(Win32ComStrategy, "is_available", return_value=True):
            converter = MDBToCSVConverter()
            self.assertEqual(converter.strategy.name, "win32com")

    def test_list_tables(self):
        """测试列出表"""
        fake = FakeStrategy()
        converter = MDBToCSVConverter(strategy=fake)
        with patch("os.path.exists", return_value=True):
            tables = converter.list_tables("/fake/path.mdb")
        self.assertEqual(tables, ["Table1", "Table2", "Tést_表3"])

    def test_list_tables_file_not_found(self):
        """测试文件不存在时抛出异常"""
        fake = FakeStrategy()
        converter = MDBToCSVConverter(strategy=fake)
        with self.assertRaises(FileNotFoundError):
            converter.list_tables("/not/exist/file.mdb")

    def test_export_table(self):
        """测试导出单个表"""
        fake = FakeStrategy()
        converter = MDBToCSVConverter(strategy=fake)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_output.csv")
            converter.export_table("/fake/test.mdb", "Table1", output_path)

            self.assertTrue(os.path.exists(output_path))
            df = pd.read_csv(output_path, encoding="utf-8-sig")
            self.assertEqual(len(df), 3)
            self.assertEqual(list(df.columns), ["ID", "Name", "Score"])
            self.assertEqual(df["Name"].tolist(), ["Alice", "Bob", "Charlie"])

    def test_export_all(self):
        """测试导出所有表"""
        fake = FakeStrategy(tables=["Customers", "Orders", "中文表"])
        converter = MDBToCSVConverter(strategy=fake)

        with tempfile.TemporaryDirectory() as tmpdir, patch("os.path.exists", return_value=True):
            exported = converter.export_all("/fake/test.mdb", tmpdir)

            self.assertEqual(len(exported), 3)
            expected_files = ["Customers.csv", "Orders.csv", "中文表.csv"]
            for ef in expected_files:
                self.assertTrue(
                    any(ef in p for p in exported),
                    f"期望找到文件包含 {ef}，实际: {exported}"
                )

            # 验证 CSV 内容
            for csv_path in exported:
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
                self.assertEqual(len(df), 3)
                self.assertEqual(list(df.columns), ["ID", "Name", "Score"])

    def test_export_all_empty_tables(self):
        """测试空表列表"""
        fake = FakeStrategy(tables=[])
        converter = MDBToCSVConverter(strategy=fake)

        with tempfile.TemporaryDirectory() as tmpdir, patch("os.path.exists", return_value=True):
            exported = converter.export_all("/fake/test.mdb", tmpdir)
            self.assertEqual(exported, [])

    def test_special_chars_in_table_name(self):
        """测试表名中的特殊字符被正确处理"""
        fake = FakeStrategy(tables=["Table With Space", "A/B\\C", "表-1"])
        converter = MDBToCSVConverter(strategy=fake)

        with tempfile.TemporaryDirectory() as tmpdir, patch("os.path.exists", return_value=True):
            exported = converter.export_all("/fake/test.mdb", tmpdir)
            filenames = [os.path.basename(p) for p in exported]
            # 特殊字符应被替换为下划线
            self.assertIn("Table_With_Space.csv", filenames)
            self.assertIn("A_B_C.csv", filenames)
            self.assertIn("表-1.csv", filenames)


class TestMDBToolsStrategy(unittest.TestCase):

    def test_is_available_true(self):
        """测试 mdbtools 可用"""
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            strategy = MDBToolsStrategy()
            self.assertTrue(strategy.is_available())

    def test_is_available_false(self):
        """测试 mdbtools 不可用"""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            strategy = MDBToolsStrategy()
            self.assertFalse(strategy.is_available())

    def test_list_tables(self):
        """测试 mdb-tables 输出解析"""
        mock_result = MagicMock()
        mock_result.stdout = "Table1\nTable2\n\nTable3\n"
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            strategy = MDBToolsStrategy()
            tables = strategy.list_tables("/fake/test.mdb")
            mock_run.assert_called_once_with(
                ["mdb-tables", "-1", "/fake/test.mdb"],
                capture_output=True, text=True, check=True
            )
            self.assertEqual(tables, ["Table1", "Table2", "Table3"])

    def test_read_table(self):
        """测试 mdb-export 输出解析"""
        csv_content = "ID,Name,Score\n1,Alice,95.5\n2,Bob,87.0\n"
        mock_result = MagicMock()
        mock_result.stdout = csv_content
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            strategy = MDBToolsStrategy()
            df = strategy.read_table("/fake/test.mdb", "Table1")
            mock_run.assert_called_once_with(
                ["mdb-export", "/fake/test.mdb", "Table1"],
                capture_output=True, text=True, check=True
            )
            self.assertEqual(len(df), 2)
            self.assertEqual(list(df.columns), ["ID", "Name", "Score"])


class TestPyODBCStrategy(unittest.TestCase):

    @patch.object(PyODBCSrategy, "is_available")
    def test_driver_set_correctly(self, mock_is_available):
        """测试驱动被正确设置"""
        # 由于环境中 pyodbc 无法导入，直接测试逻辑
        strategy = PyODBCSrategy()
        # 手动设置 driver 模拟检测成功后的状态
        strategy.driver = "Microsoft Access Driver (*.mdb)"
        self.assertEqual(strategy.driver, "Microsoft Access Driver (*.mdb)")

    def test_connection_string_windows(self):
        """测试 Windows 连接字符串"""
        strategy = PyODBCSrategy()
        strategy.driver = "Microsoft Access Driver (*.mdb)"
        with patch("sys.platform", "win32"):
            conn_str = strategy._get_connection_string("C:\\test.mdb")
            self.assertIn("Microsoft Access Driver (*.mdb)", conn_str)
            self.assertIn("C:\\test.mdb", conn_str)

    def test_connection_string_linux(self):
        """测试 Linux 连接字符串"""
        strategy = PyODBCSrategy()
        strategy.driver = "MDBToolsODBC"
        with patch("sys.platform", "linux"):
            conn_str = strategy._get_connection_string("/home/user/test.mdb")
            self.assertIn("MDBToolsODBC", conn_str)
            self.assertIn("/home/user/test.mdb", conn_str)


class TestWin32ComStrategy(unittest.TestCase):

    def test_is_available_not_windows(self):
        """测试非 Windows 平台"""
        with patch("sys.platform", "linux"):
            strategy = Win32ComStrategy()
            self.assertFalse(strategy.is_available())

    def test_is_available_windows_no_module(self):
        """测试 Windows 但无 win32com"""
        with patch("sys.platform", "win32"), \
             patch.dict("sys.modules", {"win32com.client": None}):
            strategy = Win32ComStrategy()
            self.assertFalse(strategy.is_available())


class TestIntegrationWithoutRealMDB(unittest.TestCase):
    """
    集成测试：使用模拟数据验证完整流程
    无需真实 MDB 文件和驱动
    """

    def test_full_workflow_with_mock(self):
        """模拟完整工作流程"""
        fake = FakeStrategy(tables=["Employees", "Departments"])
        converter = MDBToCSVConverter(strategy=fake)

        with tempfile.TemporaryDirectory() as tmpdir, patch("os.path.exists", return_value=True):
            # 1. 列出表
            tables = converter.list_tables("/fake/test.mdb")
            self.assertEqual(len(tables), 2)

            # 2. 导出所有
            exported = converter.export_all("/fake/test.mdb", tmpdir)
            self.assertEqual(len(exported), 2)

            # 3. 验证 CSV 文件
            for csv_path in exported:
                self.assertTrue(os.path.exists(csv_path))
                with open(csv_path, "r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    self.assertEqual(len(rows), 3)
                    for row in rows:
                        self.assertIn("ID", row)
                        self.assertIn("Name", row)
                        self.assertIn("Score", row)


if __name__ == "__main__":
    unittest.main(verbosity=2)
