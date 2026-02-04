from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import time


def extract_steps_from_left_pane(driver, wait_time=2):
    time.sleep(wait_time)
    lis = driver.find_elements(By.XPATH, "//*[@id='leftPane']//li")

    result = []
    for li in lis:
        try:
            img_tag = li.find_element(By.TAG_NAME, "img")
            img = img_tag.get_attribute("src")
        except:
            img = None

        text = li.text.strip()
        result.append({"text": text, "img": img})

    return result


def extract_steps_from_right_pane(driver, wait_time=2):
    wait = WebDriverWait(driver, 30)
    iframe = driver.find_element(By.CSS_SELECTOR, "#judge-comment iframe")
    driver.switch_to.frame(iframe)
    wait.until(lambda d: d.execute_script("return document.readyState") == "complete")

    judge_comment = driver.find_element(By.XPATH, "/html/body/p").text

    li_elements = driver.find_elements(By.TAG_NAME, "li")

    result = []
    for li in li_elements:
        try:
            img_tag = li.find_element(By.TAG_NAME, "img")
            img = img_tag.get_attribute("src")
        except:
            img = None

        p = li.find_element(By.TAG_NAME, "p")
        text = p.text.strip()
        result.append({"text": text, "img": img})

    driver.switch_to.default_content()
    return (judge_comment, result)
