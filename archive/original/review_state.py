"""Embed durable, attributed review decisions in the offline comparison page."""
import hashlib
import html
import json
import re
from pathlib import Path


def integrate_review(page, config, summary, out):
    source = out / 'dictated-feedback.json'
    if not source.exists():
        return page
    feedback = json.loads(source.read_text())
    if feedback['run_sha256'] != summary['run_sha256']:
        raise ValueError('Feedback belongs to another experiment')
    rejected = {'036_pair', '055_pair', '056_composite', '057_composite', '058_composite'}
    rows = []
    for original in feedback['decisions']:
        row = dict(original)
        # Keep the spoken record intact; status normalisation is explicitly attributed.
        row['spoken_verdict'] = original['verdict']
        row['verdict'] = ('reject' if row['id'] in rejected else
                          'good_baseline' if row['id'] == '041_pair' else 'needs_improvement')
        row['verdict_basis'] = 'Status interpreted from the user comment; editable.'
        row['preference_basis'] = ('Explicit user comparison.' if row['preference'] != 'not_rated'
                                   else 'The user did not explicitly name a winner.')
        row['winner'] = {'reference_better': 'ChatGPT', 'baseline_better': config['model']['id'],
                         'similar': 'tie'}.get(row['preference'])
        if row['id'] == '041_pair':
            row['verdict_basis'] = 'User described this result as generally good, with improvements needed.'
        row['shared_feedback'] = feedback['general_feedback']['notes']
        rows.append(row)
    state = dict(feedback, version=2, decisions=rows,
                 labels={'reference': 'ChatGPT — approved example; exact model unknown',
                         'baseline': config['model']['id'] + ' — no added LoRA'})
    state['feedback_revision'] = hashlib.sha256(source.read_bytes()).hexdigest()
    (out / 'review-decisions.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    lookup = {r['id']: r for r in rows}
    esc = lambda x: html.escape(str(x), quote=True)

    def card(match):
        text = match.group(0)
        ident = re.search(r'data-id="([^"]+)"', text).group(1)
        row = lookup[ident]
        text = text.replace('<article ', '<article id="' + esc(ident) + '" ', 1)
        for field in ('verdict', 'preference'):
            pattern = r'(<select class="' + field + r'"[^>]*>)(.*?)(</select>)'
            def select(m):
                options = re.sub(r'\sselected(?:="[^"]*")?', '', m.group(2))
                options = options.replace('value="' + row[field] + '"',
                                          'value="' + row[field] + '" selected')
                return m.group(1) + options + m.group(3)
            text = re.sub(pattern, select, text, flags=re.S)
        text = re.sub(r'<input class="review-note"[^>]*>',
                      lambda _: '<label class="note-label">Review comments<textarea class="review-note" rows="4">'
                      + esc(row['notes']) + '</textarea></label>', text)
        origin = ('Shared feedback applied at your request' if row['source'] == 'user_authorised_shared_feedback'
                  else 'Your dictated feedback · translated into English')
        badge = '<p class="feedback-origin">' + origin + ' · <span class="verdict-basis">' + esc(row['verdict_basis']) + '</span></p>'
        return text.replace('<div class="controls">', badge + '<div class="controls">', 1)

    page = re.sub(r'<article\b.*?</article>', card, page, flags=re.S)
    model_name = config['model']['id'].split('/')[-1]
    page = page.replace('<h3>Approved example</h3>', '<h3>ChatGPT · approved example</h3>')
    page = page.replace('Your approved example</figcaption>', 'Generated in ChatGPT · exact model unknown</figcaption>')
    page = page.replace('<h3>Base model</h3>', '<h3>' + esc(model_name) + '</h3>')
    page = page.replace('Qwen base model · no added LoRA', esc(model_name) + ' · no added LoRA')
    page = page.replace('Example is better</option>', 'ChatGPT wins</option>')
    page = page.replace('Base model is better</option>', esc(model_name) + ' wins</option>')
    page = page.replace('Compared with the example ', 'Winner ')
    page = re.sub(r'<title>.*?</title>', '<title>ChatGPT vs ' + esc(model_name) + ' · pencil review</title>', page)
    page = re.sub(r'<h1>.*?</h1>', '<h1>Original → ChatGPT → ' + esc(model_name) + '</h1>', page)
    shared = ('<section class="shared-feedback"><h2>Saved review feedback</h2><p>'
              'Your comments and explicit winners are embedded in this HTML and saved in '
              '<a href="review-decisions.json">review-decisions.json</a>. '
              'Verdicts are interpreted from your comments and labelled as such. '
              'An unnamed winner remains not rated. Browser edits override the saved starting values.</p>'
              '<details><summary>Shared criteria applied across the review</summary><p>'
              + esc(feedback['general_feedback']['notes']) + '</p></details></section>')
    page = page.replace('<main>', '<main>' + shared, 1)
    page = re.sub(r'<p class="notice">.*?</p>', '<p class="notice">All dictated feedback is loaded automatically. '
                  'Edit a verdict, winner or comment below; export your changes to save them to a file. '
                  '<span id="status-note"></span></p>', page, flags=re.S)
    page = page.replace('</style>', '.review-note{display:block;min-height:110px;resize:vertical;line-height:1.5}'
                        '.note-label{display:block;margin-top:12px}.feedback-origin{font-size:13px;color:#536578;'
                        'padding:9px;background:#f0f5fa;border-radius:6px}.shared-feedback{background:#eaf2f7;'
                        'padding:16px;border-radius:10px;margin-top:18px}.shared-feedback h2{margin:0;font-size:18px}'
                        '.shared-feedback p{margin:8px 0;font-size:14px}</style>')
    payload = json.dumps(state, ensure_ascii=False).replace('<', '\\u003c')
    page = re.sub(r'<script>.*?</script>', lambda _: '<script id="review-data" type="application/json">'
                  + payload + '</script><script src="review.js?v=' + state['feedback_revision'][:12] + '"></script>', page, flags=re.S)
    (out / 'review.js').write_text((Path(__file__).with_name('review.js')).read_text())
    return page
