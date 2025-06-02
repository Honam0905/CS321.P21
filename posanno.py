import re
import gradio as gr
from underthesea import word_tokenize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Embedded PRD data for training
TEXTT_PRD = '''
((S-HLN (NP-SBJ (NONE *E*)) (VP (Vv-H Thêm) (NP-DOB (Nn-H cơ_hội)) (PP-IOB (Cs-H cho) (NP (Nn-H bệnh_nhân) (VP (Vv-H ghép) (NP (Nn-H thận) (PU ,) (Nn-H gan)))))) (PU .)))
((S (NP-SBJ (Nn-H Bệnh_viện) (Nr Chợ_Rẫy) (LBRK LBRK) (NP-PRN-LOC (Nun-H TP) (Nr HCM)) (RBRK RBRK)) (VP (R đang) (Vv-H nghiên_cứu) (Vv-H ứng_dụng) (VP (Vv-H ghép) (NP (Nn-H thận)) (PP (Cs-H từ) (NP (Nn-H người) (Vv cho) (SBAR (Cs *0*) (S (NP-SBJ (Nn-H tim)) (VP (Vv-H ngừng) (VP-CMP (Vv-H đập))))))))) (PU ,) (PP-PRP (Cs-H nhằm) (VP (VP (Vv-H mở_rộng) (NP (Nn-H nguồn) (VP (Vv-H cho) (NP (Nn-H tạng))))) (PU ,) (VP (Vv-H đem) (R lại) (VP (Vv-H thêm)) (NP-DOB (Nn-H cơ_hội) (VP (Vv-H được) (VP-CMP (Vv-H ghép) (NP-DOB (Nn-H thận))))) (PP-IOB (Cs-H cho) (NP (Nn-H bệnh_nhân)))))) (PU .)))
((S (NP-SBJ-1 (Nn-H Thông_tin) (Nn trên)) (VP (Vv-H được) (S-CMP (NP-SBJ (Nn_w (Sv Phó) (Nn-H giáo_sư)) (PU -) (Nn-H tiến_sĩ) (PU -) (Nn-H bác_sĩ) (Nr Nguyễn_Trường_Sơn) (PU ,) (NP-PRN (Nn-H Giám_đốc) (NP (Nn-H Bệnh_viện) (Nr Chợ_Rẫy))) (PU ,)) (VP (Vv-H cho) (NP-IOB (NONE *E*)) (VP-CMP (Vv-H biết) (NP-CMP (NONE *T*-1)) (PP (Cs-H trong) (NP (Nn-H-2 Hội_nghị) (Nn Khoa_học) (ADJP (Aa-H thường_niên)) (NP-TMP (Num-H 2014)) (VP (Vv-H được) (VP-CMP (Vv-H tổ_chức) (NP-DOB (Nn *D*-2)) (PP-LOC (Cs-H tại) (NP (Nn-H Bệnh_viện) (Nr Chợ_Rẫy))) (NP-TMP (Nt-H hôm_nay) (NP-PRN (Nt-H 18.4))))))))))) (PU .)))
((S (VP-MNR (Vv-H Theo) (NP (Nn-H bác_sĩ) (Nr Sơn))) (PU ,) (S-SBJ (NP-SBJ-1 (Nn-H nghiên_cứu) (Pd này)) (VP (Vv-H được) (VP-CMP (Vv-H đưa) (NP-DOB (NONE *T*-1)) (PP (Cs-H vào) (VP (Vv-H thực_hiện)))))) (VP (R sẽ) (Vv-H góp_phần) (VP-CMP (Vv-H giải_quyết) (NP-DOB (QP (R hơn) (Num-H 5.000)) (Nn-H-2 trường_hợp) (VP (VP (Vv-H suy) (NP (Nn-H thận)) (ADJP (Aa-H mạn_tính)) (NP (Nn-H giai_đoạn) (Aa cuối))) (PU ,) (VP (R đang) (Vv-H phải) (VP-CMP (VP (Vv-H lọc) (NP (Nn-H màn) (Nn bụng)) (SBAR-PRP (NONE *-1))) (Cp hoặc) (VP (Vv-H chạy) (NP (Nn-H thận) (ADJP (Aa-H nhân_tạo))) (SBAR-PRP-1 (Cs vì) (S (NP-SBJ (Nn *H*-2)) (VP (R không) (Ve-H có) (NP-CMP (Nn-H thận)) (PP-PRP (Cs-H để) (VP (Vv-H ghép))))))))))))) (PU .)))
((S (ADJP-ADV (Aa-H Song_song) (NP (Pd-H đó))) (PU ,) (NP-SBJ (Nq các) (Nn-H bác_sĩ) (PP (Cs-H tại) (NP (Nn-H Bệnh_viện) (Nr Chợ_Rẫy)))) (VP (R cũng) (Vv-H nghiên_cứu) (NP-DOB (Nn-H đề_tài) (VP (Vv-H phòng_ngừa) (Cp và) (Vv-H điều_trị) (NP (Nq một_số) (Nn-H vi_rút) (PP (Cs-H trong) (VP (Vv-H ghép) (NP (Nn-H thận)))))))) (PU .)))
((S (NP-TMP (Nn-H Thời_gian) (ADJP (Aa-H qua))) (PU ,) (NP-SBJ-1 (Num một) (Nn-H nghiên_cứu) (Nn khoa_học) (Aa khác) (VP (Vv-H liên_quan) (PP (Cs-H đến) (VP (Vv-H ghép) (NP (Nn-H tạng)))))) (VP (RP (R cũng) (R-H đang)) (Vv-H được) (VP-CMP (Vv-H thực_hiện) (NP-DOB (NONE *T*-1)) (PP-LOC (Cs-H tại) (NP (Nn-H Bệnh_viện) (Nr Chợ_Rẫy))))) (PU .)))
((S (NP-SBJ (Pd-H Đó)) (VP (Vc-H là) (NP-CMP (Nn-H ứng_dụng) (VP (Vv-H ghép) (NP (Nn-H gan)) (VP-MNR (Vv-H theo) (NP (NP (Nn-H mô_hình) (NP (Nn-H người) (VP (Vv-H sống)) (VP (Vv-H cho) (NP (Nn-H tạng))))) (Cp và) (NP (Nn-H mô_hình) (NP (Nn-H người) (NP (Nn-H chết_não)) (VP (Vv-H hiến) (NP (Nn-H tạng)))))))))) (PU .)))
((S (PP-TMP (Cs-H Trong) (NP (Num hai) (Nt-H năm) (NP (Num-H 2012) (Cp và) (Num-H 2013)))) (PU ,) (NP-SBJ (Nn-H Bệnh_viện) (Nr Chợ_Rẫy)) (VP (R đã) (Vv-H tiến_hành) (NP-CMP (Num hai) (Nn-H ca) (VP (Vv-H ghép) (NP (Nn-H gan) (NP (Nn-H người_lớn))) (PP (Cs-H từ) (NP (Nn-H người) (Vv cho) (VP (Vv-H sống))))))) (PU .)))
((S (PP-ADV (Cs-H Trong) (NP (Pd-H đó))) (PU ,) (NP-SBJ (Nn-H trường_hợp) (Aa đầu)) (VP (Vv-H tử_vong) (NP-TMP (Num 2) (Nu-H tháng) (PP (Cs-H sau) (VP (Vv-H mổ))))) (PU .)))
((S (NP-TPC (Nn-H Trường_hợp) (Vv ghép) (NP (Nn-H thứ) (An hai))) (PU ,) (PP-TMP (Cs-H đến) (NP (Nt-H nay))) (PU ,) (S (NP-SBJ (Nn-H sức_khỏe) (NP-1 (Nn-H bệnh_nhân))) (VP (Vv-H phục_hồi))) (PU ,) (S (NP-SBJ (Nn-H chức_năng) (NP (Nn-H gan))) (UCP-PRD (ADJP (Aa-H ổn_định)) (Cp và) (VP (Vv-H tiến_triển) (ADJP (Aa-H tốt)) (PP-TMP (Cs-H sau) (NP (QP (R hơn) (Num-H 7)) (Nu-H tháng) (SBAR (Cs *0*) (S (NP-SBJ-2 (NONE *-1)) (VP (Vv *P*) (VP-CMP (Vv-H phẫu_thuật) (NP-DOB (NONE *T*-2))))))))))) (PU .)))
((S (NP-SBJ (Nn-H Bác_sĩ) (Nr Sơn)) (VP (Vv-H hi_vọng) (SBAR-CMP (Cs *0*) (S (S-SBJ (NP-SBJ (Nq các) (Nn-H nghiên_cứu) (Pd này)) (VP (Vv-H thành_công))) (VP (R sẽ) (Vv-H mở) (R ra) (VP (Vv-H thêm)) (NP-DOB (ADJP (Aa-H nhiều)) (Nn-H cơ_hội)) (PP-IOB (Cs-H cho) (NP (Nq các) (Nn-H bệnh_nhân) (VP (Vv-H cần) (VP-CMP (Vv-H ghép) (NP (Nn-H tạng)))))) (PP (Cs-H trong) (NP (Nn-H điều_kiện) (SBAR (Cs *0*) (S (NP-SBJ (Nn-H nguồn) (VP (Vv-H hiến) (NP (Nn-H tạng)))) (ADJP-PRD (R rất) (Aa-H hiếm_hoi)))) (PP (Cs-H như) (NP (Nt-H hiện_nay))))))))) (PU .)))
((S (PP-ADV (Cs-H Tại) (NP (Nn-H-1 Hội_nghị) (Nn Khoa_học) (ADJP (Aa-H thường_niên)) (VP (Vv-H được) (VP-CMP (Vv-H tổ_chức) (NP-DOB (Nn *D*-1)) (PP-LOC (Cs-H tại) (NP (Nn-H Bệnh_viện) (Nr Chợ_Rẫy))))) (NP (Nn-H lần) (Pd này)))) (PU ,) (NP-SBJ-3 (ADJP (Aa-H nhiều)) (NP (NP (Nn-H nghiên_cứu) (Nn khoa_học) (PP (NONE *-2))) (PU ,) (NP (Nn-H ứng_dụng) (Aa mới) (PP-2 (Cs-H trong) (VP (Vv-H điều_trị) (NP (Nn-H y_khoa))))))) (VP (R cũng) (Vv-H được) (VP-CMP (Vv-H báo_cáo) (PU ,) (Vv-H công_bố) (NP-DOB (NONE *T*-1)))) (PU .)))
'''

# POS tag set from original code
LABEL_LIST = [
    'Sv', 'Nc', 'Ncs', 'Nu', 'Nun', 'Nw', 'Num', 'Nq', 'Nr', 'Nt', 'Nn',
    'Ve', 'Vc', 'D', 'VA', 'VN', 'NA', 'Vcp', 'Vv', 'An', 'Aa', 'Pd', 'Pp',
    'R', 'Cs', 'Cp', 'ON', 'ID', 'E', 'M', 'FW', 'X', 'PU'
]

class VietnameseKNNPOSTagger:
    def __init__(self, n_neighbors=12, label_list=None):
        self.n_neighbors = n_neighbors
        self.label_list = label_list or LABEL_LIST
        self.model = None
        self.label_encoder = LabelEncoder()
        self.vectorizer = DictVectorizer(sparse=False)
        self.accuracy = None

    def _extract_features(self, word, index=None, sentence=None):
        feats = {
            'word.lower()': word.lower(),
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'has_hyphen': '-' in word,
            'length': len(word)
        }
        if sentence is not None:
            if index > 0:
                prev = sentence[index-1]
                feats.update({
                    'prev.lower()': prev.lower(),
                    'prev.istitle()': prev.istitle(),
                    'prev.isupper()': prev.isupper()
                })
            if index < len(sentence)-1:
                nxt = sentence[index+1]
                feats.update({
                    'next.lower()': nxt.lower(),
                    'next.istitle()': nxt.istitle(),
                    'next.isupper()': nxt.isupper()
                })
        return feats

    @staticmethod
    def _remove_starred(text):
        return re.sub(r"\*[-\w]+\*", "", text)

    @staticmethod
    def _remove_brackets(text):
        return text.replace('(', ' ').replace(')', ' ').replace('LBRK',' ').replace('RBRK',' ')

    @staticmethod
    def _clean_prd(text):
        lines = []
        for ln in text.splitlines():
            ln = VietnameseKNNPOSTagger._remove_starred(ln)
            ln = VietnameseKNNPOSTagger._remove_brackets(ln)
            ln = ln.strip()
            if ln:
                lines.append(ln.split())
        return lines

    def train(self):
        raw_lines = self._clean_prd(TEXTT_PRD)
        X, y = [], []
        for tokens in raw_lines:
            for i, tok in enumerate(tokens):
                parts = tok.split('-')
                label = parts[0] if parts[0] in self.label_list else None
                if label and i+1 < len(tokens):
                    word = tokens[i+1]
                    X.append(self._extract_features(word, i, tokens))
                    y.append(label)
        # Encode and vectorize
        y_enc = self.label_encoder.fit_transform(y)
        X_vec = self.vectorizer.fit_transform(X)
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_vec, y_enc, test_size=0.2, random_state=42
        )
        # Train KNN
        clf = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        clf.fit(X_train, y_train)
        self.model = clf
        # Evaluate
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        print(f"POS tagger accuracy: {self.accuracy:.4f}")

    def predict(self, tokens):
        feats = [self._extract_features(tok, i, tokens) for i, tok in enumerate(tokens)]
        X_vec = self.vectorizer.transform(feats)
        y_enc = self.model.predict(X_vec)
        tags = self.label_encoder.inverse_transform(y_enc)
        return tags

# Train tagger
tagger = VietnameseKNNPOSTagger()
print("Training POS tagger...")
tagger.train()
print("Training complete.")

# Gradio interface

def pos_annotate(raw_sentence):
    token_str = word_tokenize(raw_sentence, format='text')
    tokens = token_str.split()
    tags = tagger.predict(tokens)
    annotated = ' '.join(f"{t}/{w}" for w, t in zip(tokens, tags))
    acc_text = f"Model accuracy (PRD test set): {tagger.accuracy:.4f}"
    return annotated, acc_text

iface = gr.Interface(
    fn=pos_annotate,
    inputs=gr.Textbox(lines=2, placeholder="Nhập câu tiếng Việt..."),
    outputs=[
        gr.Textbox(label="POS Annotation"),
        gr.Textbox(label="Accuracy Metric")
    ],
    title="POSAnno",
    description="Nhập câu Tiếng Việt, click Submit để nhận kết quả POS và độ chính xác của mô hình."
)

if __name__ == "__main__":
    iface.launch()
