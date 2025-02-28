# streamlit_cartpole

failure deployment as streamlit app

```

TypeError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/streamlit_cartpole/app.py", line 350, in <module>
    test_agent(policy_net)
File "/mount/src/streamlit_cartpole/app.py", line 270, in test_agent
    anim = visualize_cartpole(best_states)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/mount/src/streamlit_cartpole/app.py", line 185, in visualize_cartpole
    action = policy_net(torch.FloatTensor(state)).argmax().item()
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


```
