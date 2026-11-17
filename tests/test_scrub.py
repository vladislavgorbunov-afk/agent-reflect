from agent_reflect.scrub import scrub


def test_redacts_github_tokens():
    assert "ghp_" + "a" * 36 not in scrub("t ghp_" + "a" * 36)
    assert "[REDACTED]" in scrub("github_pat_" + "b" * 30)


def test_redacts_sk_proj_and_glpat_tokens():
    sk_proj = "sk-proj-" + "a" * 30
    assert sk_proj not in scrub("key " + sk_proj)
    glpat = "glpat-" + "a" * 20
    assert glpat not in scrub("tok " + glpat)


def test_kebab_case_words_not_mistaken_for_keys():
    s = "plan task-based-navigation-refactoring-plan, mask-image-with-long-name"
    assert scrub(s) == s


def test_bearer_value_redacted_word_kept():
    assert scrub("Authorization: Bearer abc.def-123456") == "Authorization: Bearer [REDACTED]"


def test_shell_substitution_kept():
    s = 'curl -H "Authorization: Bearer $(cat $SP/.tok_b)"'
    assert scrub(s) == s


def test_password_values():
    assert scrub("password: hunter2!") == "password: [REDACTED]"
    assert scrub("пароль hunter2") == "пароль [REDACTED]"


def test_pwd_word_kept():
    assert scrub("pwd && ls") == "pwd && ls"


def test_env_and_json_keyword_values_redacted():
    assert scrub("GH_TOKEN=abc123xyz") == "GH_TOKEN=[REDACTED]"
    assert scrub("export GITHUB_TOKEN=abc123xyz") == "export GITHUB_TOKEN=[REDACTED]"
    assert scrub("SECRET_KEY=abc123") == "SECRET_KEY=[REDACTED]"
    assert scrub('{"password": "hunter2"}') == '{"password": "[REDACTED]"}'
    assert scrub('"token": "abc123"') == '"token": "[REDACTED]"'


def test_tokenizer_word_kept():
    assert scrub("tokenizer: x") == "tokenizer: x"


def test_concatenated_keys_redacted():
    assert scrub("ACCESSTOKEN=abc123xyz") == "ACCESSTOKEN=[REDACTED]"
    assert scrub("export AUTHTOKEN=abc123") == "export AUTHTOKEN=[REDACTED]"
    assert scrub("XSECRET: hunter2") == "XSECRET: [REDACTED]"


def test_quoted_values_redacted_whole():
    assert scrub('password="my secret"') == 'password="[REDACTED]"'
    assert scrub('{"token": "a b c"}') == '{"token": "[REDACTED]"}'
    assert scrub("TOKEN: 'a b'") == "TOKEN: '[REDACTED]'"


def test_separator_does_not_cross_newline():
    assert scrub("token:\nnext") == "token:\nnext"
    assert scrub("password\nhunter2") == "password\nhunter2"


def test_slack_token_redacted():
    tok = "xoxb-" + "1234567890-abcdef"
    assert scrub("slack " + tok) == "slack [REDACTED]"


def test_token_and_secret_need_separator():
    assert scrub("token count: 5") == "token count: 5"
    assert scrub("token abc123") == "token abc123"
    assert scrub("token: abc123") == "token: [REDACTED]"
    assert scrub("secret=abc123") == "secret=[REDACTED]"


def test_otp_and_keys():
    assert "482913" not in scrub("OTP code 482913")
    assert "sk-ant-api03-xxxxxxxxxxxx" not in scrub("key sk-ant-api03-xxxxxxxxxxxx")
    assert "AKIA" + "A" * 16 not in scrub("AKIA" + "A" * 16)
    jwt = "eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12
    assert jwt not in scrub(jwt)


def test_plain_text_untouched():
    for s in ["Exit code 137", "Timeout 30000ms exceeded", "flutter analyze failed"]:
        assert scrub(s) == s
