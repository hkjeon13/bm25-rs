from bm25 import BM25

tokenized_texts = [
    ('doc1', ["나는", "밥을", "먹는다"]),
    ('doc2', ["[서울=뉴시스]김경문", "인턴", "기자", "=", "청나라", "마지막", "황제인", "선통제", "푸이가", "착용했던", "손목시계가", "82억원에", "팔렸다.", "23일",
        "홍콩", "사우스차이나모닝포스트(SCMP),", "프랑스", "AFP통신", "등에", "따르면",
        "중국", "청나라의", "마지막", "황제", "푸이가", "생전", "찼던", "명품", "파텍필립", "손목시계가", "이날", "홍콩에서", "진행된", "필립스", "아시아", "지부",
        "경매에서", "수수료", "포함", "4890만홍콩달러(약", "82억원)에", "거래됐다.",
        "이", "시계는", "'파텍필립", "레퍼런스", "96", "콴티엠", "룬'으로", "당초", "외신은", "이", "시계가", "300만달러(약", "40억원)가", "넘는", "가격에",
        "낙찰될", "것으로", "전망했다."
    ]),
    ('doc3', [
        '국정원·검찰은', '23일', '진보당', '전', '공동대표(서울)와', '전교조', '강원지부장(강원)을', '소위', "'창원간첩단'의", '하부조직원으로', '보고', '압수수색을', '벌였다.',
        '이에', "'국가보안법", '폐지', '국민행동,', "공안탄압저지대책위'는", '24일', '서울', '서대문구', '경찰청', '앞에서', '기자회견을', '열어', '입장을', '밝혔다.',
        '기자회견은', '안지중', '국가보안법폐지국민행동', '공동운영위원장의', '사회로', '진행됐다.', '한충목', '국가보안법폐지국민행동', '공동대표,', '박승렬', '목사(NCCK인권센터',
        '부이사장),', '홍희진', '진보당', '공동대표,', '문병모', '전교조', '서울지부', '통일위원장,', '하원오', '전국농민회총연맹', '의장은', '발언을', '통해', '"공안탄압',
        '압수수색을', '규탄한다"고', '했다.'
    ])
]

bm25 = BM25()
bm25.add_documents(tokenized_texts)