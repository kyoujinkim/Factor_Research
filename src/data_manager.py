import ast
import sqlite3
import json


class MasterDataManager:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.commit_counter = 0
        self.cursor = self.conn.cursor()
        # 테이블 생성: 'section'은 대분류 key, 'data'는 중첩된 dict(JSON 형태)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS master_table (
                section TEXT PRIMARY KEY,
                data TEXT
            )
        ''')
        self.conn.commit()

    # 1. 특정 키(section) 전체 데이터 가져오기
    def get_section(self, section):
        self.cursor.execute('SELECT data FROM master_table WHERE section = ?', (section,))
        result = self.cursor.fetchone()
        try:
            val = json.loads(result[0])
        except:
            try:
                val = ast.literal_eval(result[0])
            except:
                val = result[0]
        return val if result else None

    def check_type(self, value, datatype):
        if datatype == int:
            return int(float(value))
        elif datatype == float:
            return float(value)
        elif datatype == bool:
            return bool(value)
        elif datatype == str:
            return str(value)
        else:
            return value

    def get_sub_value(self, section, sub_key, datatype=str):
        """
        특정 section 내의 특정 sub_key 값만 가져옵니다.
        """
        # json_extract(컬럼명, '$.경로') 형식을 사용합니다.
        query = f"SELECT json_extract(data, '$.{sub_key}') FROM master_table WHERE section = ?"

        self.cursor.execute(query, (section,))
        result = self.cursor.fetchone()

        if result and result[0] is not None:
            # 결과가 문자열 형태의 JSON(dict/list)일 수 있으므로 변환 시도
            try:
                return json.loads(result[0])
            except:
                # 단일 숫자나 문자열인 경우 그대로 반환
                try:
                    return ast.literal_eval(result[0])
                except:
                    val = result[0]
                    try:
                        return self.check_type(val, datatype)
                    except:
                        return val

        return None

    # 2. 특정 section 내부의 특정 sub_key 값만 수정하기 (핵심!)
    def update_sub_value(self, section, sub_key, new_value):
        # 만약 상위 섹션이 없다면 생성
        self.cursor.execute('''
        INSERT OR IGNORE INTO master_table (section, data) VALUES (?, ?)
        ''', (section, '{}'))
        # json_set 함수를 사용하면 DB 내에서 해당 부분만 수정함
        query = f"UPDATE master_table SET data = json_set(data, '$.{sub_key}', json(?)) WHERE section = ?"
        # value가 dict나 list라면 json 문자열로 변환, 단일 값이면 그대로 입력
        try:
            param_value = json.dumps(new_value) if isinstance(new_value, (dict, list)) else new_value
            self.cursor.execute(query, (param_value, section))
        except:
            param_value = json.dumps(new_value)
            self.cursor.execute(query, (param_value, section))

        self.commit()

    # 3. 새로운 section 추가 혹은 전체 덮어쓰기
    def upsert_section(self, section, dict_data):
        self.cursor.execute('''
            INSERT INTO master_table (section, data) VALUES (?, ?)
            ON CONFLICT(section) DO UPDATE SET data = excluded.data
        ''', (section, json.dumps(dict_data)))
        self.commit()

    # 5. 모든 section 키 목록 가져오기
    def get_all_sections(self)->list:
        # 모든 section 컬럼의 값을 가져오는 쿼리
        self.cursor.execute('SELECT section FROM master_table')

        # fetchall()은 [(section1,), (section2,)] 형태로 반환하므로 리스트로 변환
        results = self.cursor.fetchall()
        return [row[0] for row in results]

    def get_all_values_by_key(self, sub_key)->dict:
        query = f"SELECT section, json_extract(data, '$.{sub_key}') FROM master_table"
        self.cursor.execute(query)
        results = self.cursor.fetchall()

        values = {}
        for section, raw_json in results:
            try:
                val = json.loads(raw_json)
            except:
                try:
                    val = ast.literal_eval(raw_json)
                except:
                    val = raw_json
            values[section] = val

        return values

    def commit(self, force=False):
        if self.commit_counter >= 100 or force:
            self.conn.commit()
            self.commit_counter = 0
        else:
            self.commit_counter += 1

    def close(self):
        self.commit(force=True)
        self.conn.close()

if __name__ == "__main__":

    # --- 사용 예시 ---
    db = MasterDataManager('/equity_master.db')

    equity_master = json.load(open('/equity_master.json', 'r'))

    # 초기 데이터 저장
    initial_data = {"sub_key1": "value1", "sub_key2": 100, "sub_key3": {"nested_key": 10}}
    db.upsert_section("Factor_A", initial_data)

    # 'Factor_A'에서 'sub_key2'의 값만 200으로 수정
    db.update_sub_value("Factor_A", "sub_key2", 200)
    db.update_sub_value("Factor_A", "sub_key3.nested_key", 15)

    # 결과 확인
    print(db.get_section("Factor_A"))
    print(db.get_sub_value("Factor_A", "sub_key2"))  # 출력: 200
    print(db.get_sub_value("Factor_A", "sub_key3.nested_key"))  # 출력:
    # 출력: {'sub_key1': 'value1', 'sub_key2': 200}

    db.close()