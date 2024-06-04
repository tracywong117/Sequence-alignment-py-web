from flask import Flask, render_template, request
import torch
from wtforms import Form, TextAreaField, SubmitField, validators

app = Flask(__name__)

class InputForm(Form):
    seq1 = TextAreaField('Sequence 1:', default='A T G C', validators=[validators.InputRequired()])
    seq2 = TextAreaField('Sequence 2:', default = 'A G C T', validators=[validators.InputRequired()])
    submit = SubmitField('Align')

def global_alignment_batch_no_gap_penalty(seqs1, seqs2, device, match_score=1, mismatch_penalty=0):
    batch_size, max_len = seqs1.size()
    dtype = torch.float32
    score_matrix = torch.zeros((batch_size, max_len + 1, max_len + 1), dtype=dtype, device=device)
    match_mask = seqs1[:, :, None] == seqs2[:, None, :]
    scores_update = torch.where(match_mask, match_score, mismatch_penalty)
    for i in range(1, max_len + 1):
        for j in range(1, max_len + 1):
            match = score_matrix[:, i-1, j-1] + scores_update[:, i-1, j-1]
            delete = score_matrix[:, i-1, j]
            insert = score_matrix[:, i, j-1]
            score_matrix[:, i, j] = torch.max(torch.max(match, delete), insert)
    final_scores = score_matrix[:, -1, -1]
    return final_scores

@app.route('/', methods=['GET', 'POST'])
def index():
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        seq1 = form.seq1.data.upper().replace(" ", "").strip().split('\r\n')
        print(seq1)
        seq2 = form.seq2.data.upper().replace(" ", "").strip().split('\r\n')
        print(seq2)
        # Convert strings to tensor
        tensor_seq1 = torch.tensor([[ord(c) for c in seq] for seq in seq1], dtype=torch.long, device='cpu')
        tensor_seq2 = torch.tensor([[ord(c) for c in seq] for seq in seq2], dtype=torch.long, device='cpu')
        # Calculate alignment
        score = global_alignment_batch_no_gap_penalty(tensor_seq1, tensor_seq2, device='cpu')
        return render_template('results.html', form=form, score=score)
    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)