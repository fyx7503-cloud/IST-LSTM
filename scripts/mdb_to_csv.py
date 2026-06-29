#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MDB to CSV Converter

将 Microsoft Access MDB 文件导出为 CSV 文件。
支持多种底层策略自动检测和切换：
  1. pyodbc  — 通过 ODBC 驱动连接（推荐 Windows）
  2. mdbtools — 命令行工具（推荐 Linux/macOS）
  3. win32com — 通过 Access COM 接口（仅限 Windows）

依赖安装说明：
  - 通用:  pip install pyodbc pandas
  - Linux: sudo apt-get install mdbtools  (备选方案)
  - Windows: 安装 Microsoft Access Database Engine

用法示例：
  python mdb_to_csv.py input.mdb --output-dir ./csv_output
  python mdb_to_csv.py input.mdb --table 表名 --output output.csv
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)


class MDBStrategy:
    """策略基类"""
    name: str = "base"

    def is_available(self) -> bool:
        raise NotImplementedError

    def list_tables(self, mdb_path: str) -> List[str]:
        raise NotImplementedError

    def read_table(self, mdb_path: str, table_name: str) -> pd.DataFrame:
        raise NotImplementedError


class PyODBCSrategy(MDBStrategy):
    """使用 pyodbc + ODBC 驱动读取 MDB"""
    name = "pyodbc"

    def __init__(self):
        self.conn_str = None

    def is_available(self) -> bool:
        try:
            import pyodbc
            # 测试是否能列出驱动
            drivers = pyodbc.drivers()
            # 查找 Access 相关的驱动
            access_drivers = [d for d in drivers if 'access' in d.lower() or 'mdb' in d.lower()]
            if access_drivers:
                self.driver = access_drivers[0]
                return True
            # 在 Linux 上尝试 MDBTools ODBC 驱动
            mdb_drivers = [d for d in drivers if 'mdb' in d.lower()]
            if mdb_drivers:
                self.driver = mdb_drivers[0]
                return True
            return False
        except ImportError:
            return False

    def _get_connection_string(self, mdb_path: str) -> str:
        mdb_abs = os.path.abspath(mdb_path)
        if sys.platform == "win32":
            return f"DRIVER={{{self.driver}}};DBQ={mdb_abs};"
        else:
            return f"DRIVER={{{self.driver}}};DBQ={mdb_abs};UID=;PWD=;"

    def list_tables(self, mdb_path: str) -> List[str]:
        import pyodbc
        conn_str = self._get_connection_string(mdb_path)
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        tables = [row.table_name for row in cursor.tables(tableType='TABLE')]
        conn.close()
        return tables

    def read_table(self, mdb_path: str, table_name: str) -> pd.DataFrame:
        import pyodbc
        conn_str = self._get_connection_string(mdb_path)
        conn = pyodbc.connect(conn_str)
        query = f"SELECT * FROM [{table_name}]"
        df = pd.read_sql(query, conn)
        conn.close()
        return df


class MDBToolsStrategy(MDBStrategy):
    """使用 mdbtools 命令行工具读取 MDB（Linux/macOS）"""
    name = "mdbtools"

    def is_available(self) -> bool:
        try:
            subprocess.run(["mdb-tables", "--version"], capture_output=True, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    def list_tables(self, mdb_path: str) -> List[str]:
        result = subprocess.run(
            ["mdb-tables", "-1", mdb_path],
            capture_output=True, text=True, check=True
        )
        tables = [t.strip() for t in result.stdout.strip().split("\n") if t.strip()]
        return tables

    def read_table(self, mdb_path: str, table_name: str) -> pd.DataFrame:
        result = subprocess.run(
            ["mdb-export", mdb_path, table_name],
            capture_output=True, text=True, check=True
        )
        # mdb-export 输出 CSV 格式
        lines = result.stdout.strip().split("\n")
        if not lines:
            return pd.DataFrame()
        reader = csv.reader(lines)
        rows = list(reader)
        if not rows:
            return pd.DataFrame()
        headers = rows[0]
        data = rows[1:]
        df = pd.DataFrame(data, columns=headers)
        return df


class Win32ComStrategy(MDBStrategy):
    """使用 Windows COM 接口通过 Access 应用打开（仅 Windows）"""
    name = "win32com"

    def is_available(self) -> bool:
        if sys.platform != "win32":
            return False
        try:
            import win32com.client
            return True
        except ImportError:
            return False

    def list_tables(self, mdb_path: str) -> List[str]:
        import win32com.client
        engine = win32com.client.Dispatch("DAO.DBEngine.36")
        db = engine.OpenDatabase(os.path.abspath(mdb_path))
        tables = []
        for t in db.TableDefs:
            if not t.Name.startswith("MSys"):  # 跳过系统表
                tables.append(t.Name)
        db.Close()
        return tables

    def read_table(self, mdb_path: str, table_name: str) -> pd.DataFrame:
        import win32com.client
        engine = win32com.client.Dispatch("DAO.DBEngine.36")
        db = engine.OpenDatabase(os.path.abspath(mdb_path))
        rs = db.OpenRecordset(f"SELECT * FROM [{table_name}]")
        columns = [rs.Fields(i).Name for i in range(rs.Fields.Count)]
        data = []
        while not rs.EOF:
            row = [rs.Fields(i).Value for i in range(rs.Fields.Count)]
            data.append(row)
            rs.MoveNext()
        rs.Close()
        db.Close()
        return pd.DataFrame(data, columns=columns)


class MDBToCSVConverter:
    """MDB 到 CSV 转换器，自动选择最佳策略"""

    def __init__(self, strategy: Optional[MDBStrategy] = None):
        self.strategy = strategy or self._auto_detect_strategy()

    @staticmethod
    def _auto_detect_strategy() -> MDBStrategy:
        strategies = [PyODBCSrategy(), MDBToolsStrategy(), Win32ComStrategy()]
        for s in strategies:
            if s.is_available():
                print(f"[INFO] 使用策略: {s.name}")
                return s
        raise RuntimeError(
            "未找到可用的 MDB 读取策略。\n"
            "请安装以下依赖之一:\n"
            "  - Windows: pip install pyodbc (需安装 Access Database Engine)\n"
            "  - Linux:   sudo apt-get install mdbtools\n"
            "  - 或使用:  pip install pyodbc (需安装 unixODBC + libmdbodbc)"
        )

    def list_tables(self, mdb_path: str) -> List[str]:
        if not os.path.exists(mdb_path):
            raise FileNotFoundError(f"MDB 文件不存在: {mdb_path}")
        return self.strategy.list_tables(mdb_path)

    def export_table(self, mdb_path: str, table_name: str, output_path: str) -> None:
        df = self.strategy.read_table(mdb_path, table_name)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"  ✓ 导出: {table_name} -> {output_path} ({len(df)} 行, {len(df.columns)} 列)")

    def export_all(self, mdb_path: str, output_dir: str) -> List[str]:
        tables = self.list_tables(mdb_path)
        if not tables:
            print("[WARN] MDB 文件中未找到用户表")
            return []
        os.makedirs(output_dir, exist_ok=True)
        exported = []
        print(f"[INFO] 发现 {len(tables)} 个表，开始导出...")
        for table in tables:
            safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in table)
            output_path = os.path.join(output_dir, f"{safe_name}.csv")
            self.export_table(mdb_path, table, output_path)
            exported.append(output_path)
        print(f"[INFO] 导出完成: {len(exported)} 个文件 -> {output_dir}")
        return exported


def main():
    parser = argparse.ArgumentParser(
        description="将 Microsoft Access MDB 文件导出为 CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s database.mdb                    # 导出所有表到默认目录
  %(prog)s database.mdb -o ./output        # 指定输出目录
  %(prog)s database.mdb -t 客户表 -c 客户.csv  # 导出指定表
  %(prog)s database.mdb --list             # 仅列出所有表名
        """
    )
    parser.add_argument("mdb_file", help="输入的 MDB 文件路径")
    parser.add_argument("-o", "--output-dir", default="./csv_output",
                        help="输出目录 (默认: ./csv_output)")
    parser.add_argument("-t", "--table", help="仅导出指定表名")
    parser.add_argument("-c", "--csv-file", help="指定输出 CSV 文件名（配合 -t 使用）")
    parser.add_argument("-e", "--encoding", default="utf-8-sig",
                        help="输出编码 (默认: utf-8-sig)")
    parser.add_argument("--list", action="store_true",
                        help="仅列出 MDB 中的所有表名")

    args = parser.parse_args()

    try:
        converter = MDBToCSVConverter()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if args.list:
        print("[INFO] MDB 文件中的表:")
        for t in converter.list_tables(args.mdb_file):
            print(f"  - {t}")
        return

    if args.table:
        output_path = args.csv_file or f"{args.table}.csv"
        converter.export_table(args.mdb_file, args.table, output_path)
    else:
        converter.export_all(args.mdb_file, args.output_dir)


if __name__ == "__main__":
    main()
