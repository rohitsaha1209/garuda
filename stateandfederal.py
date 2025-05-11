from playwright.sync_api import sync_playwright

def login_and_save_state(
    login_url="https://www.stateandfederalbids.com/bids/myAccount",
    username="your_email@example.com",
    password="your_password",
    storage_file="stateandfederal.json"
):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        
        # Go to login page
        page.goto(login_url)

        # Fill in login form — update selectors based on actual form
        page.fill("input[name='user']", username)
        page.fill("input[name='password']", password)
        page.click("input[value='login']")

        # Wait for navigation or confirmation that login succeeded
        page.wait_for_load_state("networkidle")

        # Save login/session state
        context.storage_state(path=storage_file)
        browser.close()



def use_logged_in_session(url, storage_file="stateandfederal.json"):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(storage_state=storage_file)
        page = context.new_page()

        page.goto(url)
        page.wait_for_timeout(5000)
        content = page.content()
        browser.close()
        return content

