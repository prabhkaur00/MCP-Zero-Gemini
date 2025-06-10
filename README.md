## MCP-Zero: Proactive Toolchain Construction for LLM Agents from Scratch

<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
  <img src="assets/robot.png" alt="MCP-Zero Robot" width="24" height="24">
  <a href="https://arxiv.org/abs/2506.01056">
    <img src="https://img.shields.io/badge/arXiv-2506.01056-red" height="20">
  </a>
</div>

Paper: [https://arxiv.org/abs/2506.01056](https://arxiv.org/abs/2506.01056)


Aha, you find the repo of MCP-Zero so early! :D

We've released our dataset and partial code, while more features are on the way. Please check it out!


<div align="center">
  <img src="assets/fig1.png" alt="MCP-Zero workflow">
  <p> Using MCP-Zero to proactively construct toolchains for "Making a great meal"</p>
</div>


### Method: MCP-Zero

- **File Path**: `./MCP-zero/`

We have now released our code for hierarchical semantic matching, and other features will be added recently. Leave a star🌟 to let me know you are staying updated :D



### Dataset: MCP-tools

- **Dataset Path**: `./MCP-tools/mcp_tools_with_embedding.json`

- **Google Drive**: [Download Link](https://drive.google.com/file/d/1RjBGU-AGdHdhUABoeYSztbfQlD0hjUBn/view?usp=sharing)

- **Huggingface Link**: Coming soon

> My Git LFS bandwidth quota has been exhausted, please use the Google Drive link to download instead. Thank you all for your attention to this work. 


**Introduction**: A dataset containing all filtered tools from the MCP official repo. 308 servers and 2,797 tools in total.

**Data structure**:
```
{
  "server_name": string, // The name of the MCP server, extracted or inferred from the README
  "server_summary": string, // A summary of the server's purpose and capabilities, based on all relevant parts of the README.
  "server_description": string, // Description from metadata. 
  "description_embedding": float[3072], // The embedding of the server description from text-embedding-3-large
  "summary_embedding": float[3072], // The embedding of the server summary from text-embedding-3-large
  "tools": [
    {
      "name": string, // The function/tool name
      "description": string, // A concise description of what the tool does
      "description_embedding": float[3072], // The embedding of the tool description from text-embedding-3-large
      "parameter": { // A dictionary of input parameters, being included if explicitly defined
        "param1": "(type) description1",
        "param2": "(Optional, type) description2"
      }
    }
  ]
}
```


### Citation

> Citation makes me happy.
> 
>   --Shakespeare
>   ~~(just for fun :D)~~

```bibtex
@article{fei2025mcp,
  title={MCP-Zero: Proactive Toolchain Construction for LLM Agents from Scratch},
  author={Fei, Xiang and Zheng, Xiawu and Feng, Hao},
  journal={arXiv preprint arXiv:2506.01056},
  year={2025}
}
```

